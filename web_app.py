import os
import time
from datetime import datetime

import cv2
import cvzone
import numpy as np
import pygame
from cvzone.FaceMeshModule import FaceMeshDetector
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(filename):
    return os.path.join(BASE_DIR, filename)


def format_duration(total_seconds):
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


app = Flask(__name__, template_folder="templates")
monitor_active = False
last_warning = "Monitoring stopped"
cap = None
study_seconds = 0.0
last_frame_time = time.time()
current_day = datetime.now().date().isoformat()

pygame.mixer.init()

alarm_sleep = pygame.mixer.Sound(resource_path("alarm.mp3"))
alarm_facehide = pygame.mixer.Sound(resource_path("faudio.mp3"))
alarm_phone = pygame.mixer.Sound(resource_path("paudio.mp3"))

current_playing = None


def open_camera():
    global cap
    if cap is not None and cap.isOpened():
        return cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


face_detector = FaceMeshDetector(maxFaces=1)
phone_detector = YOLO(resource_path("yolov8n.pt"))
classNames = phone_detector.names

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
FACE_LEFT = 130
FACE_RIGHT = 243

closed_frames = 0
SLEEP_THRESHOLD_FRAMES = 15
covered_frames = 0
COVER_THRESHOLD_FRAMES = 20


@app.route("/")
def index():
    return render_template("index.html", monitor_active=monitor_active)


def reset_day_if_needed():
    global current_day, study_seconds
    today = datetime.now().date().isoformat()
    if today != current_day:
        current_day = today
        study_seconds = 0.0


@app.route("/status")
def status():
    reset_day_if_needed()
    return jsonify({
        "status": "started" if monitor_active else "stopped",
        "warning": last_warning,
        "study_time": format_duration(study_seconds),
    })


@app.route("/toggle_monitor/<action>")
def toggle_monitor(action):
    global monitor_active, cap, last_warning, study_seconds, current_day

    reset_day_if_needed()

    if action == "start":
        monitor_active = True
        last_warning = "Monitoring started"
        open_camera()
    elif action == "stop":
        monitor_active = False
        last_warning = "Monitoring stopped"
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        if cap is not None:
            cap.release()
            cap = None

    return jsonify({
        "status": "started" if monitor_active else "stopped",
        "warning": last_warning,
        "study_time": format_duration(study_seconds),
    })


def detect_and_draw(img):
    global current_playing, closed_frames, covered_frames, last_warning, study_seconds, last_frame_time

    reset_day_if_needed()
    elapsed = time.time() - last_frame_time
    last_frame_time = time.time()

    is_audio_busy = pygame.mixer.get_busy()
    if not is_audio_busy:
        current_playing = None

    img, faces = face_detector.findFaceMesh(img, draw=False)
    is_sleepy = False
    is_face_covered = False

    if faces:
        covered_frames = 0
        face = faces[0]

        eye_dist, _ = face_detector.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
        face_dist, _ = face_detector.findDistance(face[FACE_LEFT], face[FACE_RIGHT])
        ratio = (eye_dist / face_dist) * 100

        if ratio < 11.0:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames >= SLEEP_THRESHOLD_FRAMES:
            is_sleepy = True

        cvzone.putTextRect(img, f"Eye Ratio: {int(ratio)}", (30, 40), scale=1, thickness=1)
    else:
        closed_frames = 0
        covered_frames += 1
        if covered_frames >= COVER_THRESHOLD_FRAMES:
            is_face_covered = True

    results = phone_detector.predict(img, stream=True, verbose=False)
    phone_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if classNames[cls_id] == "cell phone" and conf > 0.5:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cvzone.putTextRect(img, f"Phone detected! {int(conf * 100)}%", (x1, max(y1 - 10, 30)), scale=1, thickness=1, colorR=(255, 0, 255))

    if is_face_covered:
        warning = "DONT COVER YOUR FACE!"
        if not is_audio_busy:
            alarm_facehide.play(0)
            current_playing = "facehide"
        last_warning = warning
    elif is_sleepy:
        warning = "WAKE UP & STUDY!"
        if not is_audio_busy:
            alarm_sleep.play(0)
            current_playing = "sleep"
        last_warning = warning
    elif phone_detected:
        warning = "PUT THE PHONE AWAY!"
        if not is_audio_busy:
            alarm_phone.play(0)
            current_playing = "phone"
        last_warning = warning
    elif is_audio_busy:
        if current_playing == "facehide":
            warning = "DONT COVER YOUR FACE!"
        elif current_playing == "sleep":
            warning = "WAKE UP & STUDY!"
        elif current_playing == "phone":
            warning = "PUT THE PHONE AWAY!"
        if 'warning' in locals():
            last_warning = warning
    else:
        if current_playing is not None:
            pygame.mixer.stop()
            current_playing = None
        warning = "Monitoring active"
        last_warning = warning
        study_seconds += elapsed

    return img


def generate_frames():
    while True:
        if not monitor_active:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Monitoring Stopped", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        if cap is None or not cap.isOpened():
            open_camera()

        success, img = cap.read()
        if not success:
            break
        processed = detect_and_draw(img)
        ret, buffer = cv2.imencode('.jpg', processed)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
