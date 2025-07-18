# 🎯 RealTimePostureCorrect

**RealTimePostureCorrect** is a real-time posture monitoring and correction system that uses computer vision to analyze body positioning from a live camera feed. The system detects incorrect posture patterns and provides immediate feedback, helping users develop healthy sitting habits—especially valuable for remote workers, students, and professionals.

---

## 🧠 Project Overview

This project was developed as part of an academic assignment focused on practical applications of real-time image processing and AI-assisted feedback systems. The core objective is to monitor and classify human posture in real time using MediaPipe landmarks and machine learning.

---

## 🚀 Key Features

- 🧍‍♂️ **Real-Time Pose Detection** – Leverages MediaPipe for accurate skeleton tracking.
- 📊 **Posture Classification** – Uses a trained ML model to differentiate between correct and incorrect postures.
- 🔔 **Instant Feedback** – Alerts the user visually or audibly when incorrect posture is detected.
- 📈 **Posture Logging** – Optionally logs posture status over time for review or analytics.
- ⚙️ **Custom Thresholds** – Configurable parameters to define what constitutes "bad posture".
- 💡 **Lightweight and Efficient** – Runs in real-time on standard consumer hardware.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Libraries**: 
  - MediaPipe
  - OpenCV
  - NumPy
  - scikit-learn / PyTorch (for model training and inference)
  - matplotlib (for data visualization, optional)

---

## 🖥️ System Architecture

1. **Camera Feed** → 
2. **MediaPipe Pose Detection** → 
3. **Feature Extraction (angles, keypoints)** → 
4. **ML Classifier (Good/Bad Posture)** → 
5. **Feedback Mechanism (Alert / GUI)**

---

## 📦 Installation

### Prerequisites

- Python 3.10 or newer
- A working webcam
- Git

### Clone and Setup

```bash
git clone https://github.com/harn0ld/RealTimePostureCorrect.git
cd RealTimePostureCorrect
pip install -r requirements.txt
