from pathlib import Path

import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
from collections import deque
import pygame

DEFAULT_CAMERA_INDEX = 1


class PostureMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Posture Monitor System")
        self.root.geometry("1000x700")

        # Initialize pygame for sound alerts
        pygame.mixer.init()
        base_path = Path("D:/PyCharm Community Edition 2023.1.1/jbr/pythonProject2")
        self.model = joblib.load(base_path / "rf_model.pkl")
        self.scaler = joblib.load(base_path / "scaler.pkl")
        # Load model and scaler
        try:
            base_path = Path("D:/PyCharm Community Edition 2023.1.1/jbr/pythonProject2")
            self.model = joblib.load(base_path / "rf_model.pkl")
            self.scaler = joblib.load(base_path / "scaler.pkl")

        except:
            messagebox.showerror("Error", "Could not load model files (rf_model.pkl, scaler.pkl)")
            return

        self.CLASS_MAP = {
            0: "looks good",
            1: "sit up straight"
        }

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

        # Monitoring variables
        self.monitoring = False
        self.cap = None
        self.alert_threshold = 20.0  # seconds of bad posture in 30s window
        self.window_duration = 30.0  # seconds
        self.posture_history = deque()  # (timestamp, is_bad_posture)
        self.last_alert_time = 0
        self.alert_cooldown = 10.0  # seconds between alerts

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.start_btn = ttk.Button(control_frame, text="Start Monitoring", command=self.toggle_monitoring)
        self.start_btn.grid(row=0, column=0, padx=(0, 10))

        ttk.Label(control_frame, text="Alert Threshold (seconds):").grid(row=0, column=1, padx=(10, 5))
        self.threshold_var = tk.DoubleVar(value=self.alert_threshold)
        threshold_spin = ttk.Spinbox(control_frame, from_=5.0, to=30.0, increment=1.0,
                                     width=10, textvariable=self.threshold_var,
                                     command=self.update_threshold)
        threshold_spin.grid(row=0, column=2, padx=(0, 10))

        # Status panel
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.status_label = ttk.Label(status_frame, text="Status: Not monitoring", font=("Arial", 12))
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        self.posture_label = ttk.Label(status_frame, text="Posture: Unknown", font=("Arial", 12))
        self.posture_label.grid(row=1, column=0, sticky=tk.W)

        self.bad_posture_label = ttk.Label(status_frame, text="Bad posture time: 0.0s / 30s", font=("Arial", 12))
        self.bad_posture_label.grid(row=2, column=0, sticky=tk.W)

        # Progress bar for bad posture time
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, length=300, variable=self.progress_var, maximum=30.0)
        self.progress_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        self.video_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.video_label = ttk.Label(self.video_frame, text="Camera feed will appear here")
        self.video_label.grid(row=0, column=0)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

    def update_threshold(self):
        self.alert_threshold = self.threshold_var.get()

    def toggle_monitoring(self):
        if not self.monitoring:
            self.start_monitoring()
        else:
            self.stop_monitoring()

    def start_monitoring(self):
        try:
            self.cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return

            self.monitoring = True
            self.start_btn.config(text="Stop Monitoring")
            self.status_label.config(text="Status: Monitoring active")
            self.posture_history.clear()

            # Start video processing thread
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")

    def stop_monitoring(self):
        self.monitoring = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(text="Start Monitoring")
        self.status_label.config(text="Status: Not monitoring")
        self.video_label.config(image="", text="Camera feed will appear here")

    def extract_features(self, landmarkss):
        landmarks = np.array(landmarkss, dtype=np.float32)

        # Obsługa confidence
        confidence_score = np.mean(landmarks[:, 2]) if landmarks.shape[1] > 2 else 1.0

        # Współrzędne 2D
        landmarks_2d = landmarks[:, :2]

        # Kluczowe punkty
        nose = landmarks_2d[0]
        left_eye = landmarks_2d[2]
        right_eye = landmarks_2d[5]
        left_ear = landmarks_2d[7]
        right_ear = landmarks_2d[8]
        left_shoulder = landmarks_2d[9]
        right_shoulder = landmarks_2d[10]
        left_elbow = landmarks_2d[11]
        right_elbow = landmarks_2d[12]

        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_eyes = (left_eye + right_eye) / 2
        horiz = np.array([1.0, 0.0])
        vertical = np.array([0.0, -1.0])

        def angle_between(v1, v2):
            v1 = v1 / (np.linalg.norm(v1) + 1e-8)
            v2 = v2 / (np.linalg.norm(v2) + 1e-8)
            return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder) + 1e-8

        # Cechy kątowe
        angle_feats = [
            angle_between(right_shoulder - left_shoulder, nose - mid_shoulder),
            angle_between(right_shoulder - left_shoulder, horiz),
            angle_between(nose - mid_shoulder, vertical),
            angle_between(left_elbow - left_shoulder, horiz),
            angle_between(right_elbow - right_shoulder, horiz),
            np.linalg.norm(right_shoulder - left_shoulder) / 2.0,
            (left_shoulder[1] - right_shoulder[1]) / shoulder_width,
            (nose[0] - mid_shoulder[0]) / shoulder_width,
        ]

        selected_points = [nose, left_eye, right_eye, left_ear, right_ear,
                           left_shoulder, right_shoulder, left_elbow, right_elbow, mid_shoulder]
        coords = np.concatenate(selected_points) / shoulder_width

        dists = np.array([
            np.linalg.norm(nose - left_shoulder),
            np.linalg.norm(nose - right_shoulder),
            np.linalg.norm(left_eye - left_shoulder),
            np.linalg.norm(right_eye - right_shoulder),
            np.linalg.norm(left_shoulder - left_elbow),
            np.linalg.norm(right_shoulder - right_elbow),
        ]) / shoulder_width

        # Zaawansowane cechy geometryczne
        ear_height_diff = (left_ear[1] - right_ear[1]) / shoulder_width
        symmetry = (np.linalg.norm(left_shoulder - left_elbow) - np.linalg.norm(
            right_shoulder - right_elbow)) / shoulder_width
        eye_ear_height_diff = (mid_eyes[1] - ((left_ear[1] + right_ear[1]) / 2)) / shoulder_width
        shoulder_elbow_diff_L = (left_shoulder[1] - left_elbow[1]) / shoulder_width
        shoulder_elbow_diff_R = (right_shoulder[1] - right_elbow[1]) / shoulder_width
        head_angle = angle_between(nose - mid_eyes, mid_eyes - mid_shoulder)
        eye_horizontal_diff = (left_eye[0] - right_eye[0]) / shoulder_width
        ear_horizontal_diff = (left_ear[0] - right_ear[0]) / shoulder_width
        neck_height = np.linalg.norm(nose - mid_shoulder) / shoulder_width
        shoulder_to_neck_ratio = shoulder_width / (neck_height + 1e-8)
        head_centering = abs(nose[0] - mid_shoulder[0]) / shoulder_width

        extra_features = np.array([
            symmetry,
            ear_height_diff,
            eye_ear_height_diff,
            shoulder_elbow_diff_L,
            shoulder_elbow_diff_R,
            head_angle,
            eye_horizontal_diff,
            ear_horizontal_diff,
            shoulder_to_neck_ratio,
            head_centering,
            confidence_score
        ])

        feature_vector = np.concatenate([angle_feats, coords, dists, extra_features])
        return feature_vector.reshape(1, -1)

    def update_posture_history(self, is_bad_posture):
        current_time = time.time()
        self.posture_history.append((current_time, is_bad_posture))

        # Remove entries older than window_duration
        while self.posture_history and current_time - self.posture_history[0][0] > self.window_duration:
            self.posture_history.popleft()

    def calculate_bad_posture_time(self):
        if not self.posture_history:
            return 0.0

        current_time = time.time()
        bad_posture_time = 0.0

        for i in range(len(self.posture_history)):
            timestamp, is_bad = self.posture_history[i]
            if is_bad:
                # Calculate duration for this entry
                if i < len(self.posture_history) - 1:
                    next_timestamp = self.posture_history[i + 1][0]
                    duration = next_timestamp - timestamp
                else:
                    duration = current_time - timestamp
                bad_posture_time += duration

        return bad_posture_time

    def send_alert(self):
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return

        self.last_alert_time = current_time

        # Visual alert
        messagebox.showwarning("Posture Alert", "Sit up straight! You've been slouching too long.")

        # Try to play sound alert (optional)
        try:
            # You can add a sound file here
            # pygame.mixer.Sound("alert.wav").play()
            pass
        except:
            pass

    def process_video(self):
        previous_landmarks = None

        while self.monitoring:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            label = "No person detected"
            confidence = 0.0
            is_bad_posture = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                previous_landmarks = landmarks
            elif previous_landmarks is not None:
                landmarks = previous_landmarks
            else:
                self.update_posture_history(False)
                self.update_ui(frame, label, confidence, 0.0)
                continue

            try:
                data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                required_indices = [0, 2, 5, 7, 8, 9, 10]
                if any(results.pose_landmarks.landmark[i].visibility < 0.5 for i in required_indices):
                    raise ValueError("Missing or occluded key landmarks")

                features = self.extract_features(data)
                if np.any(np.isnan(features)) or np.any(np.isinf(features)) or features.shape[1] != 45:
                    raise ValueError("Invalid features")

                scaled = self.scaler.transform(features)
                probs = self.model.predict_proba(scaled)[0]
                pred = np.argmax(probs)
                conf = probs[pred]

                if conf > 0.5:
                    label = self.CLASS_MAP[pred]
                    confidence = conf
                    is_bad_posture = (pred == 1)  # "sit up straight"
                else:
                    label = "Uncertain"

                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                label = "Invalid frame"
                confidence = 0.0

            # Update posture tracking
            self.update_posture_history(is_bad_posture)
            bad_posture_time = self.calculate_bad_posture_time()

            # Check for alert condition
            if bad_posture_time >= self.alert_threshold:
                self.send_alert()

            self.update_ui(frame, label, confidence, bad_posture_time)

    def update_ui(self, frame, label, confidence, bad_posture_time):
        # Update video display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image)

        self.root.after(0, lambda: self.video_label.config(image=photo, text=""))
        self.root.after(0, lambda: setattr(self.video_label, 'image', photo))

        # Update status labels
        color = "green" if confidence > 0.5 and label == "looks good" else "red"
        self.root.after(0, lambda: self.posture_label.config(text=f"Posture: {label}", foreground=color))
        self.root.after(0,
                        lambda: self.bad_posture_label.config(text=f"Bad posture time: {bad_posture_time:.1f}s / 30s"))
        self.root.after(0, lambda: self.progress_var.set(bad_posture_time))

        # Change progress bar color based on threshold
        if bad_posture_time >= self.alert_threshold:
            self.root.after(0, lambda: self.progress_bar.config(style="red.Horizontal.TProgressbar"))
        else:
            self.root.after(0, lambda: self.progress_bar.config(style="TProgressbar"))


if __name__ == "__main__":
    root = tk.Tk()
    app = PostureMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_monitoring(), root.destroy()))
    root.mainloop()
