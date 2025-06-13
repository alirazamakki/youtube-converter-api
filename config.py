from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    download_dir: str = "downloads"
    max_concurrent: int = 5
    file_expiry: int = 10  # minutes
    min_disk_space: int = 1  # GB

    class Config:
        env_file = ".env"

settings = Settings() 