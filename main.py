import cv2
import numpy as np
import joblib
import mediapipe as mp
import time

# Wczytaj model RandomForest i scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

CLASS_MAP = {
    0: "looks good",
    1: "sit up straight"
}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

def extract_features(landmarkss):
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
    symmetry = (np.linalg.norm(left_shoulder - left_elbow) - np.linalg.norm(right_shoulder - right_elbow)) / shoulder_width
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


# Kamera
cap = cv2.VideoCapture(0)
previous_landmarks = None
frames = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    label = "No person detected"
    confidence = 0.0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        previous_landmarks = landmarks
    elif previous_landmarks is not None:
        landmarks = previous_landmarks
    else:
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Posture Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    try:
        data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        required_indices = [0, 2, 5, 7, 8, 9, 10]
        if any(results.pose_landmarks.landmark[i].visibility < 0.5 for i in required_indices):
            raise ValueError("Missing or occluded key landmarks")

        features = extract_features(data)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)) or features.shape[1] != 45:
            raise ValueError("Invalid features")

        scaled = scaler.transform(features)
        probs = model.predict_proba(scaled)[0]
        pred = np.argmax(probs)
        conf = probs[pred]
        # print("Features (scaled):", scaled)
        # print("Probs:", probs)
        if conf > 0.5:
            label = CLASS_MAP[pred]
            confidence = conf
        else:
            label = "Uncertain"

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    except Exception as e:
        label = "Invalid frame"
        confidence = 0.0
        print(f"[Warning] {e}")

    # FPS
    frames += 1
    if time.time() - start_time >= 1.0:
        print(f"[Info] FPS: {frames}")
        frames = 0
        start_time = time.time()

    # Wynik
    color = (0, 255, 0) if confidence > 0.8 else (0, 0, 255)
    cv2.putText(frame, f"Posture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Posture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import torch
# import mediapipe as mp
# from sklearn.preprocessing import StandardScaler
# import torch.nn.functional as F
# import torch.nn as nn
# import joblib
# import time
#
#
# class PoseClassifier(nn.Module):
#     def __init__(self, input_dim=39, num_classes=2):
#         super(PoseClassifier, self).__init__()
#
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.25),
#
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#
#             nn.Linear(32, num_classes)
#         )
#
#     def forward(self, x):
#         return self.net(x)
# # Załaduj model PyTorch (zdefiniuj klasę lub importuj)
# model = PoseClassifier(input_dim=39, num_classes=2)
# state_dict = torch.load('model_state_dict.pt', map_location='cpu')  # lub 'cuda' jeśli używasz GPU
# model.load_state_dict(state_dict) # ścieżka do wag
# model.eval()
#
# # Załaduj scaler (StandardScaler)
# scaler = joblib.load("scaler.pkl")
#
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_draw = mp.solutions.drawing_utils
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
#
# def angle_between(v1, v2):
#     v1 = v1.to(torch.float32)
#     v2 = v2.to(torch.float32)
#     v1 = v1 / (torch.norm(v1) + 1e-8)
#     v2 = v2 / (torch.norm(v2) + 1e-8)
#     dot = torch.dot(v1, v2)
#     return torch.acos(torch.clamp(dot, -1.0, 1.0)).item()
#
# def triangle_area(a, b, c):
#     return 0.5 * torch.abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])))
#
# def extract_features(landmarkss):
#     landmarks = torch.tensor(landmarkss, dtype=torch.float32)
#     landmarks_2d = landmarks[:, :2]
#
#     # Wybrane punkty
#     nose = landmarks_2d[0]
#     left_eye = landmarks_2d[2]
#     right_eye = landmarks_2d[5]
#     left_ear = landmarks_2d[7]
#     right_ear = landmarks_2d[8]
#     left_shoulder = landmarks_2d[9]
#     right_shoulder = landmarks_2d[10]
#     left_elbow = landmarks_2d[11]
#     right_elbow = landmarks_2d[12]
#
#     mid_shoulder = (left_shoulder + right_shoulder) / 2
#     mid_eyes = (left_eye + right_eye) / 2
#     horiz = torch.tensor([1.0, 0.0])
#     vertical = torch.tensor([0.0, -1.0])
#
#     angle_feats = [
#         angle_between(right_shoulder - left_shoulder, nose - mid_shoulder),
#         angle_between(right_eye - left_eye, nose - mid_eyes),
#         angle_between(right_shoulder - left_shoulder, horiz),
#         angle_between(nose - mid_shoulder, vertical),
#         torch.norm(right_shoulder - left_shoulder) / 2.0,
#         left_shoulder[1] - right_shoulder[1],
#         nose[0] - mid_shoulder[0],
#         angle_between(right_eye - left_eye, horiz),
#         torch.norm(right_eye - left_eye),
#         torch.norm(nose - mid_eyes),
#         angle_between(left_ear - nose, right_ear - nose)
#     ]
#     angle_feats = torch.tensor(angle_feats)
#
#     selected_points = [nose, left_eye, right_eye, left_ear, right_ear,
#                        left_shoulder, right_shoulder, left_elbow, right_elbow, mid_shoulder]
#     coords = torch.cat(selected_points).view(-1)
#
#     extra_dists = torch.tensor([
#         torch.norm(nose - left_shoulder),
#         torch.norm(nose - right_shoulder),
#         torch.norm(left_eye - left_shoulder),
#         torch.norm(right_eye - right_shoulder),
#         torch.norm(left_shoulder - left_elbow),
#         torch.norm(right_shoulder - right_elbow),
#     ])
#
#     triangle_feat = triangle_area(nose, left_shoulder, right_shoulder)
#     shoulder_len_diff = torch.norm(left_shoulder - left_elbow) - torch.norm(right_shoulder - right_elbow)
#
#     extras = torch.tensor([triangle_feat, shoulder_len_diff])
#
#     full_vector = torch.cat([angle_feats, coords, extra_dists, extras]).view(1, -1)
#     return full_vector.numpy()
#
#
#
# CLASS_MAP = {
#     0: "looks good",
#     1: "sit up straight",
# }
#
# cap = cv2.VideoCapture(0)
#
# previous_landmarks = None
#
# frames = 0
# start_time = time.time()
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb)
#
#     label = "No person detected"
#     confidence = 0.0
#
#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#         previous_landmarks = landmarks
#     elif previous_landmarks is not None:
#         landmarks = previous_landmarks
#     else:
#         cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow('Posture Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         continue
#
#     try:
#         data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
#         required_indices = [0, 2, 5, 7, 8, 9, 10]
#         if any(results.pose_landmarks.landmark[i].visibility < 0.5 for i in required_indices):
#             raise ValueError("Missing or occluded key landmarks")
#
#         features = extract_features(data)
#
#         if np.any(np.isnan(features)) or np.any(np.isinf(features)) or features.shape[1] != 39:
#             raise ValueError("Invalid features")
#
#         scaled_features = scaler.transform(features)
#         input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
#
#         with torch.no_grad():
#             logits = model(input_tensor)
#             probs = F.softmax(logits, dim=1)
#             conf, pred = torch.max(probs, dim=1)
#             label = CLASS_MAP[pred.item()]
#             confidence = conf.item()
#         print("Features (raw):", features)
#         print("Features (scaled):", scaled_features)
#         print("Probs:", probs.cpu().numpy())
#         mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#
#     except Exception as e:
#         label = "Invalid frame"
#         confidence = 0.0
#         print(f"[Warning] Skipped frame due to error: {e}")
#
#     # FPS counter (diagnostic only)
#     frames += 1
#     if time.time() - start_time >= 1.0:
#         print(f"[Info] FPS: {frames}")
#         frames = 0
#         start_time = time.time()
#
#     # Show classification result
#     cv2.putText(frame, f"Posture: {label} ({confidence*100:.1f}%)", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if confidence > 0.8 else (0, 0, 255), 2)
#
#     cv2.imshow('Posture Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break