import os
import sys
import time

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(BASE_DIR, filename)


def format_duration(total_seconds):
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def draw_warning_banner(img, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1.5
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = max(20, (img.shape[1] - text_w) // 2)
    y = 90
    pad_x = 18
    pad_y = 14

    cv2.rectangle(img, (x - pad_x, y - text_h - pad_y), (x + text_w + pad_x, y + pad_y), (20, 20, 20), -1)
    cv2.rectangle(img, (x - pad_x, y - text_h - pad_y), (x + text_w + pad_x, y + pad_y), color, 3)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def main():
    pygame.mixer.init()

    alarm_files = {
        "sleep": resource_path("alarm.mp3"),
        "facehide": resource_path("faudio.mp3"),
        "phone": resource_path("paudio.mp3"),
    }

    alarm_sleep = pygame.mixer.Sound(alarm_files["sleep"]) if os.path.exists(alarm_files["sleep"]) else None
    alarm_facehide = pygame.mixer.Sound(alarm_files["facehide"]) if os.path.exists(alarm_files["facehide"]) else None
    alarm_phone = pygame.mixer.Sound(alarm_files["phone"]) if os.path.exists(alarm_files["phone"]) else None

    current_playing = None
    study_seconds = 0.0
    last_frame_time = time.time()
    session_started = time.time()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

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

    while True:
        success, img = cap.read()
        if not success:
            break

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
            draw_warning_banner(img, "DONT COVER YOUR FACE!", (0, 0, 255))
            if not is_audio_busy and alarm_facehide is not None:
                alarm_facehide.play(0)
                current_playing = 'facehide'
        elif is_sleepy:
            draw_warning_banner(img, "WAKE UP & STUDY!", (0, 0, 255))
            if not is_audio_busy and alarm_sleep is not None:
                alarm_sleep.play(0)
                current_playing = 'sleep'
        elif phone_detected:
            draw_warning_banner(img, "PUT THE PHONE AWAY!", (0, 165, 255))
            if not is_audio_busy and alarm_phone is not None:
                alarm_phone.play(0)
                current_playing = 'phone'
        elif is_audio_busy:
            if current_playing == 'facehide':
                draw_warning_banner(img, "DONT COVER YOUR FACE!", (0, 0, 255))
            elif current_playing == 'sleep':
                draw_warning_banner(img, "WAKE UP & STUDY!", (0, 0, 255))
            elif current_playing == 'phone':
                draw_warning_banner(img, "PUT THE PHONE AWAY!", (0, 165, 255))
        else:
            study_seconds += elapsed

        study_time = format_duration(study_seconds)
        cv2.putText(img, f"Study Time: {study_time}", (img.shape[1] - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Smart Study Monitor", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
