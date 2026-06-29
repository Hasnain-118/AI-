Color Card Web Launcher 🎨
Real-time webcam app that opens websites when you show colored cards.

Quick Start
bash
pip install opencv-python numpy
python color_launcher.py
🎯 How It Works
Hold a colored card in front of your webcam for 1.5 seconds to launch:

Color	Action
🔵 Blue	Opens Facebook
🔴 Red	Opens YouTube
🟢 Green	Opens WhatsApp Web
⚙️ Customize
Edit these values in the code:

python
MIN_AREA = 8000        # Minimum card size
HOLD_SECONDS = 1.5     # Hold duration
COOLDOWN = 5.0         # Cooldown between triggers
Press Q to quit.

🛠️ Adding Colors
Add new color ranges in COLOR_RANGES and websites in WEBSITES:

python
COLOR_RANGES = {
    "purple": [(np.array([130, 50, 50]), np.array([160, 255, 255]))],
}
WEBSITES = {"purple": "https://example.com"}
📝 Requirements
Python 3.x

OpenCV

NumPy

Webcam

💡 Tips
Use solid colored cards

Ensure good lighting

Card should fill most of the frame

Made with ❤️ using OpenCV
