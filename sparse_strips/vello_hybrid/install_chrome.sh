#!/bin/bash

# Script to install Google Chrome and ChromeDriver on Ubuntu
# Usage: sudo bash install_chrome_chromedriver.sh

set -e  # Exit immediately if a command exits with a non-zero status

# Check if script is running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (with sudo)"
  exit 1
fi

# Update package lists
echo "Updating package lists..."
apt-get update

# Install dependencies
echo "Installing dependencies..."
apt-get install -y wget unzip apt-transport-https ca-certificates curl gnupg

# Add Google Chrome repository
echo "Adding Google Chrome repository..."
curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] https://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google-chrome.list

# Update package lists again after adding Chrome repository
apt-get update

# Install Google Chrome
echo "Installing Google Chrome Stable..."
apt --fix-broken install -y --no-install-recommends google-chrome-stable