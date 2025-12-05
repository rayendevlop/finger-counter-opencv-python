# finger-counter-opencv-python
I built a real-time hand-tracking and finger-counting system using OpenCV, NumPy, and classical computer-vision techniques. The project detects skin in the YCrCb color space, extracts the hand contour, computes convexity defects, and estimates the number of raised fingers — all in real time using a webcam.
1)Features
🖐 Real-time finger counting (0–5 fingers)
🎯 Skin detection in YCrCb color space
🔍 Hand contour extraction
📐 Convex hull & convexity defect analysis
🧠 Fingertip clustering for better accuracy
👥 Supports two hands
🖼 Live mask preview overlay
📸 Snapshot capture (press s)
⚡ Efficient and lightweight, runs on CPU
 2)Technologies Used
Python 3
OpenCV
NumPy
Computer Vision
Geometry-based contour analysis
_________________________
📂 Project Structure
hand-tracking-opencv/
│── hand_counter.py       # Main script
│── README.md             # Project documentation
│── snapshot.jpg          # (auto-created if you press 's')
└── requirements.txt      # Python dependencies (optional)
📦 Installation
1. Clone the repository
git clone https://github.com/rayendevlop/finger-counter-opencv-python.git
cd hand-tracking-opencv
2. Install dependencies
pip install opencv-python numpy
▶️ Run the Program
Simply execute:
python3 hand_counter.py
Controls
q or ESC → Quit
s → Save a snapshot
🧠 How It Works
1️⃣ Skin Detection (YCrCb)
A color-space threshold isolates skin pixels for better performance under different lighting.
2️⃣ Contour Extraction
cv2.findContours() retrieves hand outlines.
3️⃣ Convex Hull + Convexity Defects
Used to approximate the gaps between fingers.
4️⃣ Geometric Finger Counting
Angles + distances + fingertip clustering = reliable finger count.
📄 License
This project is licensed under the MIT License — free for personal and commercial use.
👨‍💻 Author
Rayen Gharbi
📍 Tunisia
💼 AI & Computer Vision enthusiast
