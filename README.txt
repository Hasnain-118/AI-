Vehicle Monitor System 🚗
Real-time vehicle detection + license plate recognition with email alerts.

Quick Setup
bash
pip install opencv-python numpy easyocr ultralytics torch
python vehicle_monitor.py
Configuration
python
SENDER_EMAIL    = "your@gmail.com"
RECEIVER_EMAIL  = "receiver@gmail.com"
EMAIL_PASSWORD  = "your-app-password"
VIDEO_SOURCE    = "Traffic.mp4"  # or 0 for webcam
Features
YOLOv8 vehicle detection (cars, bikes, buses, trucks)

License plate OCR with EasyOCR

Color detection (8 colors)

Email alerts for new plates

Optimized for Intel i7

Controls
Press Q to quit

Output
text
[DETECTED] Plate: ABC123 | Color: Blue
[EMAIL SENT] Plate: ABC123 | Color: Blue
Notes
Use Gmail App Password (not login password)

Video file must be in the same directory
