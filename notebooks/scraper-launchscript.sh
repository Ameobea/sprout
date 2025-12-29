# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/debian
Suites: $(. /etc/os-release && echo "$VERSION_CODENAME")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update

sudo apt install -y bzip2 docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

curl https://ameo.dev/img.tar.bz2 > img.tar.bz2
bunzip2 img.tar.bz2
sudo docker load < img.tar

sudo docker run -d --restart=always --name anime-zoo \
  -p 5700:5700 \
  -e ORIGIN="https://anime.ameo.dev" \
  -e "PORT=5700" \
  -e "DATA_DIR=/opt/data" \
  -e "MAL_CLIENT_ID=xxxalt" \
  -e "MAL_CLIENT_SECRET=xxx" \
  -e "MYSQL_HOST=nelebrie2.ameo.dev" \
  -e "MYSQL_USER=mal-collector" \
  -e "MYSQL_PASSWORD=xxx" \
  -e "MYSQL_DATABASE=anime-atlas" \
  -e "ADMIN_API_TOKEN=xxx" \
  -e "IS_DOCKER=true" \
  anime-atlas:latest
