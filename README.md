# YouTube Converter API

A FastAPI-based service for converting YouTube videos to MP3 or MP4 format.

## Features

- Convert YouTube videos to MP3 or MP4
- Real-time conversion status tracking
- Video metadata extraction
- Health monitoring
- API metrics
- Rate limiting
- File download management

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Redis (optional, for rate limiting)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-converter-api.git
cd youtube-converter-api
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
```bash
# Windows (using chocolatey)
choco install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# MacOS
brew install ffmpeg
```

5. Create a .env file:
```bash
cp .env.example .env
```

6. Configure environment variables in .env:
```env
API_HOST=0.0.0.0
API_PORT=8000
MAX_CONCURRENT_CONVERSIONS=5
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=3600
REDIS_URL=redis://localhost:6379
```

## Running the Server

1. Start the API server:
```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

2. For production deployment, use a process manager like Supervisor:
```ini
[program:youtube-converter-api]
command=/path/to/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
directory=/path/to/youtube-converter-api
user=youruser
autostart=true
autorestart=true
stderr_logfile=/var/log/youtube-converter-api.err.log
stdout_logfile=/var/log/youtube-converter-api.out.log
```

## Using Nginx as Reverse Proxy

1. Install Nginx:
```bash
# Ubuntu/Debian
sudo apt install nginx

# CentOS/RHEL
sudo yum install nginx
```

2. Create Nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Increase max upload size
    client_max_body_size 100M;
}
```

3. Enable and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/youtube-converter-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## SSL Configuration

1. Install Certbot:
```bash
sudo apt install certbot python3-certbot-nginx
```

2. Obtain SSL certificate:
```bash
sudo certbot --nginx -d your-domain.com
```

## Monitoring

The API includes Prometheus metrics at `/metrics` endpoint. You can set up monitoring using:

1. Prometheus
2. Grafana
3. Node Exporter

## Security Considerations

1. Enable rate limiting
2. Use HTTPS
3. Set up proper firewall rules
4. Monitor system resources
5. Regular security updates

## Troubleshooting

1. Check logs:
```bash
# Application logs
tail -f /var/log/youtube-converter-api.err.log
tail -f /var/log/youtube-converter-api.out.log

# Nginx logs
tail -f /var/log/nginx/error.log
tail -f /var/log/nginx/access.log
```

2. Common issues:
   - FFmpeg not found: Ensure FFmpeg is installed and in PATH
   - Permission issues: Check file permissions
   - Port conflicts: Ensure port 8000 is available
   - Memory issues: Monitor system resources

## Support

For issues and feature requests, please open an issue on GitHub.

## License

MIT License 