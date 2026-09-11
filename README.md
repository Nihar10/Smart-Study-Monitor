# Smart Study Monitor

A webcam-based study assistant designed to help students stay focused and productive while studying.

It monitors the user through the camera and detects common distractions or unhealthy study habits, including:

- face covered
- drowsiness / sleeping
- phone use during study
- looking away from the screen for too long

When one of these behaviors is detected, the app triggers a warning and plays an alarm to remind the user to refocus. The alert stops automatically once the user returns to normal behavior.

## Features

- live webcam monitoring
- start/stop controls in the browser
- real-time warning status messages
- audio alert when distraction is detected
- automatic alarm stop after recovery
- study timer tracking
- daily timer reset
- browser-based interface for local use

## Tech Stack

- Python
- OpenCV
- cvzone
- Ultralytics YOLO
- Flask
- pygame

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
python web_app.py
```

Then open:

```text
http://localhost:5000
```

## Project Purpose

This app is built for students, self-learners, and anyone who wants a simple personal study accountability system. It encourages better study habits by reducing distractions and reminding users to stay alert during study sessions.

## Notes

This project uses the webcam and computer vision model to detect user behavior in real time. It is intended for local, personal use and works best in a normal study environment with good lighting.
