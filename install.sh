#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Print with color
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root"
    exit 1
fi

# Update system
print_message "Updating system packages..."
apt update && apt upgrade -y

# Install required packages
print_message "Installing required packages..."
apt install -y python3 python3-pip python3-venv nginx redis-server ffmpeg certbot python3-certbot-nginx

# Create project directory
print_message "Creating project directory..."
mkdir -p /var/www/youtube-converter-api
cd /var/www/youtube-converter-api

# Create virtual environment
print_message "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
print_message "Installing Python requirements..."
pip install -r requirements.txt

# Create .env file
print_message "Creating environment configuration..."
cat > .env << EOL
API_HOST=0.0.0.0
API_PORT=8000
MAX_CONCURRENT_CONVERSIONS=5
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=3600
REDIS_URL=redis://localhost:6379
EOL

# Create Nginx configuration
print_message "Setting up Nginx configuration..."
cat > /etc/nginx/sites-available/tempplay.online << EOL
server {
    listen 80;
    server_name tempplay.online www.tempplay.online;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl;
    server_name tempplay.online www.tempplay.online;

    ssl_certificate /etc/letsencrypt/live/tempplay.online/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/tempplay.online/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-XSS-Protection "1; mode=block";
    add_header X-Content-Type-Options "nosniff";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }

    location /static {
        alias /var/www/youtube-converter-api/static;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    client_max_body_size 100M;
}
EOL

# Create systemd service
print_message "Creating systemd service..."
cat > /etc/systemd/system/youtube-converter.service << EOL
[Unit]
Description=YouTube Converter API
After=network.target redis.service

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/youtube-converter-api
Environment="PATH=/var/www/youtube-converter-api/venv/bin"
ExecStart=/var/www/youtube-converter-api/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Set permissions
print_message "Setting up permissions..."
chown -R www-data:www-data /var/www/youtube-converter-api
chmod -R 755 /var/www/youtube-converter-api

# Enable and start services
print_message "Starting services..."
ln -s /etc/nginx/sites-available/tempplay.online /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx

systemctl enable redis-server
systemctl start redis-server

systemctl enable youtube-converter
systemctl start youtube-converter

# Install SSL certificate
print_message "Installing SSL certificate..."
certbot --nginx -d tempplay.online -d www.tempplay.online --non-interactive --agree-tos --email admin@tempplay.online

# Setup firewall
print_message "Setting up firewall..."
apt install -y ufw
ufw allow 80
ufw allow 443
ufw --force enable

print_message "Installation completed successfully!"
print_message "You can access the API at: https://tempplay.online"
print_message "API documentation is available at: https://tempplay.online/docs"
print_message "To check the service status, run: systemctl status youtube-converter"
print_message "To view logs, run: journalctl -u youtube-converter -f" 