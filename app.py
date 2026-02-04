import os, tempfile
import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Runner Video Feature Extractor")

def calc_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

@app.get("/")
def home():
    return {"message": "Video feature service running"}

@app.post("/extract_features")
async def extract_features(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1].lower()
    if suffix not in [".mp4", ".mov", ".avi", ".mkv"]:
        return {"error": "Upload mp4/mov/avi/mkv"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.remove(video_path)
        return {"error": "Could not open video"}

    knee_angles = []
    FRAME_SKIP = 2
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % FRAME_SKIP != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            continue

        lm = results.pose_landmarks.landmark
        hip = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y)
        knee = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)
        ankle = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

        knee_angles.append(calc_angle(hip, knee, ankle))

        if len(knee_angles) >= 400:
            break

    cap.release()
    pose.close()
    os.remove(video_path)

    if len(knee_angles) < 5:
        return {"error": "Not enough pose detected (try full body visible video)"}

    avg_knee = float(np.mean(knee_angles))

    return {
        "avg_knee_angle": round(avg_knee, 2),
        "frames_used": len(knee_angles),
        "source": "video_features"
    }
