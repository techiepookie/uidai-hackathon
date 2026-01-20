#!/bin/bash

###############################################################################
# ALIS - Oracle Cloud Setup Script
# Automated deployment script for Oracle Cloud Infrastructure (OCI)
###############################################################################

set -e  # Exit on error

echo "======================================"
echo "  ALIS - Oracle Cloud Setup"
echo "======================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Please run as regular user (ubuntu), not root"
    exit 1
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install prerequisites
echo "📦 Installing prerequisites..."
sudo apt install -y curl git ufw

# Configure firewall
echo "🔒 Configuring firewall..."
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API
sudo ufw allow 8501/tcp  # Streamlit
echo "y" | sudo ufw enable

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed"
else
    echo "✅ Docker Compose already installed"
fi

# Clone repository
echo "📥 Cloning ALIS repository..."
if [ ! -d "uidai-hackathon" ]; then
    git clone https://github.com/techiepookie/uidai-hackathon.git
    cd uidai-hackathon/alis
else
    echo "✅ Repository already exists"
    cd uidai-hackathon/alis
    git pull origin real-data-implementation
fi

# Create .env file
echo "⚙️  Creating environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    
    # Get public IP
    PUBLIC_IP=$(curl -s ifconfig.me)
    
    # Update .env with public IP
    sed -i "s|BACKEND_URL=http://localhost:8000|BACKEND_URL=http://${PUBLIC_IP}:8000|g" .env
    sed -i "s|DEBUG=.*|DEBUG=false|g" .env
    sed -i "s|ENVIRONMENT=.*|ENVIRONMENT=production|g" .env
    
    echo "✅ Environment file created"
    echo "📝 Public IP: $PUBLIC_IP"
else
    echo "✅ Environment file already exists"
fi

# Build and start services
echo "🚀 Building and starting ALIS services..."
newgrp docker << END
docker-compose down 2>/dev/null || true
docker-compose build --no-cache
docker-compose up -d
END

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 15

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me)

echo ""
echo "======================================"
echo "  ✅ ALIS Deployment Complete!"
echo "======================================"
echo ""
echo "🌐 Access your application at:"
echo "   Dashboard: http://${PUBLIC_IP}:8501"
echo "   API: http://${PUBLIC_IP}:8000"
echo "   API Docs: http://${PUBLIC_IP}:8000/api/docs"
echo "   Frontend: http://${PUBLIC_IP}:80"
echo ""
echo "🔧 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop: docker-compose down"
echo "   Restart: docker-compose restart"
echo "   Update: git pull && docker-compose up -d --build"
echo ""
echo "⚠️  Note: If you see docker permission errors,"
echo "   logout and login again or run:"
echo "   newgrp docker"
echo ""
