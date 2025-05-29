# REQUIREMENTS
- Docker
- Docker compose

# Only For Ubuntu/Debian 

# Cài đặt các gói cần thiết
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Thêm GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Thêm repository
echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Cài Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Bật Docker và thêm quyền
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker

# Check
docker --version
docker-compose version

# Build
## Dừng toàn bộ container và xóa volume
docker-compose down -v
## Build và chạy lại container
docker-compose up --build

# FastAPI Docs: 
http://localhost:8000/docs