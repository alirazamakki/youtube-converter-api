import os
import uuid
import threading
from typing import Dict, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, HttpUrl, field_validator
import yt_dlp
import aiofiles
import json
import asyncio
import re
from pathlib import Path
import time
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import redis
from redis import Redis
from asyncio import Semaphore
import prometheus_client
from prometheus_client import Counter, Histogram, CollectorRegistry, Gauge
import logging
import shutil
import psutil
from redis.exceptions import RedisError
from tenacity import retry, stop_after_attempt, wait_exponential
import socket
import prometheus_client as prom
from config import settings
import httpx
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from slowapi import Limiter
from slowapi.util import get_remote_address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define metrics
metrics = {
    'REQUEST_COUNT': Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['endpoint', 'method'],
        registry=registry
    ),
    'REQUEST_LATENCY': Histogram(
        'http_request_duration_seconds',
        'HTTP request latency',
        ['endpoint'],
        registry=registry
    ),
    'CONVERSION_COUNT': Counter(
        'video_conversions_total',
        'Total video conversions',
        ['format', 'quality'],
        registry=registry
    ),
    'ERROR_COUNT': Counter(
        'api_errors_total',
        'Total API errors',
        ['endpoint', 'error_type'],
        registry=registry
    ),
    'CONVERSION_TIME': Histogram(
        'conversion_duration_seconds',
        'Time spent converting videos',
        ['format']
    ),
    'QUEUE_SIZE': Gauge(
        'conversion_queue_size',
        'Current conversion queue size'
    )
}

# Redis connection
redis_pool = redis.ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    max_connections=50,
    decode_responses=True
)
redis_conn = redis.Redis(
    connection_pool=redis_pool,
    socket_keepalive=True,
    retry_on_timeout=True
)

# Initialize FastAPI cache
FastAPICache.init(RedisBackend(redis_conn), prefix="api-cache")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, storage_uri=f"redis://{settings.redis_host}:{settings.redis_port}")

# Load allowed domains
def load_allowed_domains() -> List[str]:
    try:
        with open('allowed_domains.json', 'r') as f:
            data = json.load(f)
            return data.get('allowed_domains', [])
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Using default allowed domains")
        return ["localhost", "127.0.0.1", "example.com"]

ALLOWED_DOMAINS = load_allowed_domains()
YOUTUBE_REGEX = r"^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$"

# Constants
DOWNLOAD_DIR = Path(settings.download_dir)
DOWNLOAD_DIR.mkdir(exist_ok=True)
FILE_EXPIRY_MINUTES = settings.file_expiry
MAX_CONCURRENT_CONVERSIONS = settings.max_concurrent
RATE_LIMIT = "100/minute"
REDIS_JOB_EXPIRY = 3600  # 1 hour
METADATA_CACHE_EXPIRY = 3600  # 1 hour
MIN_DISK_SPACE_GB = settings.min_disk_space
USE_CDN = False  # Set to True if using CDN

# Check FFmpeg installation
def check_ffmpeg():
    try:
        # Check common paths on Windows first
        if os.name == 'nt':
            common_paths = [
                os.path.join(os.environ['ProgramFiles'], 'ffmpeg', 'bin', 'ffmpeg.exe'),
                os.path.join(os.environ['ProgramFiles(x86)'], 'ffmpeg', 'bin', 'ffmpeg.exe'),
                os.path.join(os.getenv('LOCALAPPDATA', ''), 'Programs', 'ffmpeg', 'bin', 'ffmpeg.exe')
            ]
            for path in common_paths:
                if os.path.exists(path):
                    logger.info(f"Found FFmpeg at: {path}")
                    return path
        
        # Fall back to system PATH
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            logger.info(f"Using FFmpeg from PATH: {ffmpeg_path}")
            return ffmpeg_path
        
        raise RuntimeError("FFmpeg not found in PATH or common locations")
    except Exception as e:
        logger.error(f"FFmpeg check failed: {str(e)}")
        raise

FFMPEG_PATH = check_ffmpeg()

def get_ydl_opts(format: str, quality: str, sanitized_title: str, video_id: str) -> dict:
    # Create unique filename to avoid conflicts
    filename = f"{sanitized_title}-{video_id}"
    filepath = str(DOWNLOAD_DIR / filename)
    
    # Common options for both formats
    common_opts = {
        'quiet': True,
        'no_warnings': True,
        'retries': 3,
        'fragment_retries': 3,
        'skip_unavailable_fragments': True,
        'hls_prefer_native': True,
        'ffmpeg_location': FFMPEG_PATH,
        'extractor_args': {
            'youtube': {
                'player_client': ['web'],
                'skip': ['hls', 'dash']
            }
        },
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'outtmpl': filepath,
        'keepvideo': False,
        'noplaylist': True,
        'concurrent_fragments': 4,
        'buffersize': 1024,
        'socket_timeout': 10,
        'extract_flat': True,
        'writethumbnail': False,
        'writesubtitles': False,
        'writeautomaticsub': False,
        'nocheckcertificate': True,
        'ignoreerrors': True,
    }
    
    if format == 'mp3':
        return {
            **common_opts,
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192' if quality == "high" else '128' if quality == "medium" else '96',
            }],
            'prefer_ffmpeg': True,
            'audioformat': 'mp3',
            'postprocessor_args': [
                '-ar', '44100',
                '-ac', '2',
                '-b:a', '192k' if quality == "high" else '128k' if quality == "medium" else '96k'
            ],
        }
    else:  # mp4
        return {
            **common_opts,
            'format': 'best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4'
            }],
            'prefer_ffmpeg': True,
            'postprocessor_args': [
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23' if quality == "high" else '28' if quality == "medium" else '32',
                '-c:a', 'aac',
                '-b:a', '192k' if quality == "high" else '128k' if quality == "medium" else '96k'
            ],
        }

async def get_video_info(url: str) -> dict:
    try:
        # Check cache first
        cache_key = f"video_info:{url}"
        cached_info = redis_conn.get(cache_key)
        if cached_info:
            return json.loads(cached_info)
        
        with yt_dlp.YoutubeDL({
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],
                    'skip': ['hls', 'dash']
                }
            },
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise HTTPException(status_code=400, detail="Failed to extract video info")
                
            video_info = {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', ''),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'id': info.get('id', '')
            }
            
            # Cache the info
            redis_conn.setex(cache_key, METADATA_CACHE_EXPIRY, json.dumps(video_info))
            
            return video_info
    except Exception as e:
        logger.error(f"Failed to get video info: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to get video info: {str(e)}")

def check_disk_space():
    """Check if there's enough disk space available."""
    try:
        free_space = psutil.disk_usage(DOWNLOAD_DIR).free
        required_space = MIN_DISK_SPACE_GB * 1024 * 1024 * 1024  # Convert GB to bytes
        return free_space >= required_space
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        return True  # Continue if we can't check disk space

# Add retry decorator for Redis operations
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def redis_operation(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except RedisError as e:
        logger.error(f"Redis operation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Redis operation: {str(e)}")
        raise

# Update the file cleanup function
async def async_cleanup_old_files():
    """Clean up files older than FILE_EXPIRY_MINUTES, excluding currently processed files."""
    try:
        current_time = datetime.now()
        lock_file = DOWNLOAD_DIR / ".cleanup.lock"
        
        # Create a lock file to prevent concurrent cleanups
        try:
            with open(lock_file, 'x') as f:
                f.write(str(os.getpid()))
        except FileExistsError:
            logger.info("Cleanup already in progress by another process")
            return
        
        try:
            # Get list of currently processed files from Redis
            processed_files = set()
            for key in redis_conn.keys("job:*"):
                job = redis_operation(redis_conn.get, key)
                if job:
                    job_data = json.loads(job)
                    if job_data.get('file_path'):
                        processed_files.add(job_data['file_path'])
            
            # Clean up old files
            for file_path in DOWNLOAD_DIR.glob("*.*"):
                try:
                    # Skip lock file and currently processed files
                    if file_path.name == ".cleanup.lock" or str(file_path) in processed_files:
                        continue
                        
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age > timedelta(minutes=FILE_EXPIRY_MINUTES):
                        logger.info(f"Removing old file: {file_path}")
                        await aiofiles.os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {str(e)}")
        finally:
            # Always remove the lock file
            try:
                lock_file.unlink()
            except Exception as e:
                logger.error(f"Error removing lock file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Update the convert_video function with better error handling
async def convert_video(token: str, url: str, format: str, quality: str):
    info = None
    try:
        # Check disk space before starting
        if not check_disk_space():
            raise RuntimeError("Not enough disk space available")
        
        # Clean up old files
        await async_cleanup_old_files()
        
        # Get video info first
        info = await get_video_info(url)
        video_id = info['id']
        sanitized_title = sanitize_filename(info['title'])
        
        if not sanitized_title:
            sanitized_title = f"video_{video_id}"
        
        # Update job status
        redis_operation(
            set_job,
            token,
            {
                'status': 'in_progress',
                'url': url,
                'format': format,
                'quality': quality,
                'file_path': None,
                'video_title': info['title']
            }
        )
        
        # Get download options
        ydl_opts = get_ydl_opts(format, quality, sanitized_title, video_id)
        base_path = DOWNLOAD_DIR / f"{sanitized_title}-{video_id}"
        
        # Add progress hooks for better logging
        def progress_hook(d):
            if d['status'] == 'downloading':
                logger.info(f"Download progress: {d.get('_percent_str', 'N/A')} of {d.get('_total_bytes_str', 'unknown size')} at {d.get('_speed_str', 'N/A')}")
            elif d['status'] == 'error':
                logger.error(f"Download error: {d}")
            elif d['status'] == 'finished':
                logger.info("Download finished, starting conversion...")
        
        ydl_opts['progress_hooks'] = [progress_hook]
        ydl_opts['logger'] = logger
        
        # Download and convert
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: ydl.download([url]))
        
        # Wait for file system to catch up
        await asyncio.sleep(2)
        
        # Find the actual file that was created
        possible_files = []
        for ext in ['.mp3', '.mp4', '.m4a', '.webm']:  # Common audio/video extensions
            possible_files.extend(DOWNLOAD_DIR.glob(f"{sanitized_title}-{video_id}*{ext}"))
            # Also try without the video ID in case it was stripped
            possible_files.extend(DOWNLOAD_DIR.glob(f"{sanitized_title}*{ext}"))
        
        if not possible_files:
            # Try finding any recently created files
            current_time = datetime.now()
            for file_path in DOWNLOAD_DIR.glob("*.*"):
                try:
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age < timedelta(minutes=1):  # Files created in the last minute
                        possible_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error checking file {file_path}: {str(e)}")
        
        if not possible_files:
            raise FileNotFoundError("Converted file not created")
            
        # Get the largest file (most likely the converted one)
        expected_path = max(possible_files, key=lambda f: f.stat().st_size)
        logger.info(f"Found converted file: {expected_path}")
        
        # Check file size
        file_size = expected_path.stat().st_size
        logger.info(f"File created: {expected_path}, Size: {file_size} bytes")
        
        min_size = 1024 * 100  # 100KB minimum
        if file_size < min_size:
            raise ValueError(f"File too small ({file_size} bytes < {min_size} bytes), likely incomplete download")
        
        # Set file expiry time
        expiry_time = datetime.now() + timedelta(minutes=FILE_EXPIRY_MINUTES)
        redis_operation(set_file_data, str(expected_path), expiry_time)
        
        # Update job status
        redis_operation(
            set_job,
            token,
            {
                'status': 'completed',
                'url': url,
                'format': format,
                'quality': quality,
                'file_path': str(expected_path),
                'video_title': info['title'],
                'file_size': file_size
            }
        )
        
        # Record metrics
        metrics['CONVERSION_COUNT'].labels(format=format, quality=quality).inc()
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}", exc_info=True)
        error_data = {
            'status': 'failed',
            'url': url,
            'format': format,
            'quality': quality,
            'error': str(e),
            'video_title': info['title'] if info else 'Unknown'
        }
        try:
            redis_operation(set_job, token, error_data)
        except Exception as redis_err:
            logger.error(f"Failed to update job status in Redis: {str(redis_err)}")
        metrics['ERROR_COUNT'].labels(endpoint="/convert", error_type="conversion_failed").inc()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(delete_expired_files())
    asyncio.create_task(adaptive_semaphore.adjust())
    yield
    # Shutdown
    redis_conn.close()

app = FastAPI(
    title="YouTube Converter API",
    lifespan=lifespan,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{domain}" for domain in ALLOWED_DOMAINS] + 
                 [f"https://{domain}" for domain in ALLOWED_DOMAINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=ALLOWED_DOMAINS
)

# Initialize Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Models
class ConversionRequest(BaseModel):
    url: HttpUrl
    format: str
    quality: str

    @field_validator('url')
    def validate_youtube_url(cls, v):
        if not re.match(YOUTUBE_REGEX, str(v)):
            raise ValueError("Invalid YouTube URL")
        return v

class VideoMetadataRequest(BaseModel):
    url: HttpUrl

    @field_validator('url')
    def validate_youtube_url(cls, v):
        if not re.match(YOUTUBE_REGEX, str(v)):
            raise ValueError("Invalid YouTube URL")
        return v

class JobStatus(BaseModel):
    status: str
    file_path: Optional[str] = None
    error: Optional[str] = None

# Helper functions
def validate_format(format: str) -> bool:
    return format.lower() in ['mp3', 'mp4']

def validate_quality(quality: str) -> bool:
    return quality.lower() in ['low', 'medium', 'high']

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '', filename).strip()

def get_job(token: str) -> Optional[dict]:
    job_data = redis_conn.get(f"job:{token}")
    return json.loads(job_data) if job_data else None

def set_job(token: str, job_data: dict):
    redis_conn.setex(f"job:{token}", REDIS_JOB_EXPIRY, json.dumps(job_data))

def get_file_data(file_path: str) -> Optional[dict]:
    key = f"file:{file_path}"
    return redis_conn.hgetall(key)

def set_file_data(file_path: str, expiry: datetime):
    key = f"file:{file_path}"
    redis_conn.hset(key, "path", file_path)
    redis_conn.hset(key, "expiry", expiry.isoformat())

def update_file_expiry(file_path: str, minutes: int = FILE_EXPIRY_MINUTES):
    key = f"file:{file_path}"
    new_expiry = datetime.now() + timedelta(minutes=minutes)
    redis_conn.hset(key, "expiry", new_expiry.isoformat())
    return new_expiry

async def delete_expired_files():
    while True:
        try:
            current_time = datetime.now()
            expired_files = []
            
            # Get all file keys
            file_keys = redis_conn.keys("file:*")
            for key in file_keys:
                file_data = redis_conn.hgetall(key)
                if file_data and 'expiry' in file_data:
                    expiry_time = datetime.fromisoformat(file_data['expiry'])
                    if current_time >= expiry_time:
                        expired_files.append((key, file_data['path']))
            
            for key, file_path in expired_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted expired file: {file_path}")
                    redis_conn.delete(key)
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
            
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in delete_expired_files: {e}")
            await asyncio.sleep(60)

# Initialize adaptive semaphore
class AdaptiveSemaphore:
    def __init__(self, initial_value):
        self.semaphore = asyncio.Semaphore(initial_value)
        self.current_limit = initial_value
        
    async def adjust(self):
        while True:
            load = psutil.cpu_percent() / 100
            new_limit = max(5, min(
                MAX_CONCURRENT_CONVERSIONS,
                int(MAX_CONCURRENT_CONVERSIONS * (1 - load))
            ))
            if new_limit != self.current_limit:
                diff = new_limit - self.current_limit
                if diff > 0:
                    for _ in range(diff):
                        self.semaphore.release()
                else:
                    for _ in range(-diff):
                        await self.semaphore.acquire()
                self.current_limit = new_limit
            await asyncio.sleep(5)

# Create adaptive semaphore instance
adaptive_semaphore = AdaptiveSemaphore(MAX_CONCURRENT_CONVERSIONS)

# Start the semaphore adjustment task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(adaptive_semaphore.adjust())

# Endpoints
@app.post("/convert")
async def create_conversion(request: ConversionRequest, background_tasks: BackgroundTasks):
    if not validate_format(request.format):
        raise HTTPException(status_code=400, detail="Invalid format. Use 'mp3' or 'mp4'")
    
    if not validate_quality(request.quality):
        raise HTTPException(status_code=400, detail="Invalid quality. Use 'low', 'medium', or 'high'")
    
    token = str(uuid.uuid4())
    
    # Set initial job status
    set_job(token, {
        'status': 'queued',
        'url': str(request.url),
        'format': request.format,
        'quality': request.quality,
        'file_path': None
    })
    
    # Start conversion
    background_tasks.add_task(
        run_conversion_with_semaphore,
        token,
        str(request.url),
        request.format,
        request.quality
    )
    
    return {"token": token, "message": "Conversion started"}

async def run_conversion_with_semaphore(token: str, url: str, format: str, quality: str):
    async with adaptive_semaphore.semaphore:
        await convert_video(token, url, format, quality)

@app.post("/video/metadata")
async def fetch_video_metadata(request: VideoMetadataRequest):
    try:
        return await get_video_info(str(request.url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status/{token}")
async def get_job_status(token: str):
    """Get conversion job status"""
    job = get_job(token)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/download/{token}")
async def download_file(token: str, request: Request):
    """Download converted file"""
    job = get_job(token)
    if not job or job['status'] != 'completed':
        raise HTTPException(status_code=404, detail="File not found or not ready")
    
    file_path = job.get('file_path')
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    if USE_CDN:
        cdn_url = f"https://cdn.example.com/downloads/{os.path.basename(file_path)}"
        return RedirectResponse(cdn_url)
    
    # Reset file expiry time on download
    update_file_expiry(file_path)
    
    return FileResponse(
        file_path,
        media_type='audio/mpeg' if job['format'] == 'mp3' else 'video/mp4',
        filename=os.path.basename(file_path),
        headers={
            'Content-Disposition': f'attachment; filename="{os.path.basename(file_path)}"'
        }
    )

@app.get("/metrics")
async def get_metrics():
    return prometheus_client.generate_latest(registry)

@app.get("/health")
async def health_check():
    """Combined health check endpoint for backward compatibility."""
    return await readiness()

@app.get("/health/liveness")
async def liveness():
    """Basic liveness check."""
    return {"status": "alive"}

@app.get("/health/readiness")
async def readiness():
    """Detailed readiness check."""
    checks = {
        "redis": redis_conn.ping(),
        "disk_space": check_disk_space(),
        "ffmpeg": bool(FFMPEG_PATH)
    }
    if all(checks.values()):
        return {"status": "ready", "services": checks}
    raise HTTPException(503, detail={"status": "not_ready", "services": checks})

if __name__ == "__main__":
    import uvicorn
    
    # Create cookies file if it doesn't exist
    if not os.path.exists('cookies.txt'):
        open('cookies.txt', 'w').close()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        timeout_keep_alive=30
    )