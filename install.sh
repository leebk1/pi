#!/bin/bash
set -e

SERVICE_NAME="telegram-bot"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/telegram_bot.py"
PYTHON_PATH="$(which python3)"
WORKING_DIR="$(cd "$(dirname "$0")" && pwd)"
USER="$(whoami)"

echo "=== Cài đặt service $SERVICE_NAME ==="
echo "  Script : $SCRIPT_PATH"
echo "  Python : $PYTHON_PATH"
echo "  User   : $USER"
echo ""

# Cài dependencies
echo "[1/4] Cài đặt dependencies..."
pip3 install requests gpiozero numpy pillow scikit-learn joblib 2>/dev/null || pip3 install --break-system-packages requests gpiozero numpy pillow scikit-learn joblib

# Tạo systemd service file
echo "[2/4] Tạo service file..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Telegram Door Lock Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORKING_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON_PATH $SCRIPT_PATH
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable và start service
echo "[3/4] Enable service..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}.service

echo "[4/4] Start service..."
sudo systemctl start ${SERVICE_NAME}.service

echo ""
echo "=== Hoàn tất! ==="
echo ""
echo "Các lệnh hữu ích:"
echo "  sudo systemctl status $SERVICE_NAME   # Xem trạng thái"
echo "  sudo journalctl -u $SERVICE_NAME -f   # Xem log realtime"
echo "  sudo systemctl restart $SERVICE_NAME   # Khởi động lại"
echo "  sudo systemctl stop $SERVICE_NAME      # Dừng bot"
