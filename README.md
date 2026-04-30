# Live-Object-Detection-Tracing
Live Object Detection & Tracking App

A real-time AI-powered web application built with Streamlit, YOLOv8, and WebRTC that detects and tracks objects through your webcam.

🚀 Features
🎯 Real-time object detection using YOLOv8
📍 Object tracking with persistent detection
⚠️ Custom alert system for selected objects
💾 Option to save detected frames
📷 Live webcam streaming in browser
⚡ Fast and lightweight inference

🛠️ Tech Stack
Python
Streamlit
Ultralytics YOLOv8
OpenCV
streamlit-webrtc
AV (PyAV)

📦 Installation
1. Install dependencies
pip install streamlit streamlit-webrtc ultralytics opencv-python av

▶️ Run the App
streamlit run app.py

⚙️ How It Works
The webcam stream is captured using WebRTC
Each frame is passed into a YOLOv8 model
The model detects and tracks objects in real-time
Results are drawn on the video stream

Optional features:
Save frames as images
Trigger alerts for specific objects

⚙️ Settings
You can control the app using the sidebar:

Save Detected Frames → Saves snapshots when objects are detected
Alert Object → Enter an object name (e.g., "person") to trigger alerts
📁 Project Structure
├── app.py
├── requirements.txt
└── README.md
⚠️ Notes
Ensure your camera is allowed in the browser
First run may take time as YOLO model downloads
Works best on Chrome or Edge browsers
🔥 Future Improvements
📊 Live analytics dashboard
📈 Detection history logging
🔊 Sound alerts
🧠 Multi-object tracking improvements
☁️ Streamlit Cloud deployment optimization
👨‍💻 Author

Built as a real-time AI computer vision project using YOLOv8 + Streamlit.
