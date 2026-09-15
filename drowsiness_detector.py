"""
Driver Drowsiness Detection System
Uses OpenCV + CNN (via dlib/MediaPipe) to detect drowsiness in real-time.
Triggers alarm popup + buzzer sound when eyes are closed for too long.

Requirements:
    pip install opencv-python mediapipe numpy pygame scipy
"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import sys

# ─── Try to import pygame for audio ───────────────────────────────────────────
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not installed. Audio alarm disabled. Run: pip install pygame")

# ─── MediaPipe Face Mesh Setup ─────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe landmark indices for eyes
# Left eye (from driver's perspective, right eye in mirror)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ─── Constants ─────────────────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.22   # Eye Aspect Ratio below this = eye closed
CONSEC_FRAMES     = 20     # Frames eye must be closed to trigger alarm (~0.67s at 30fps)
ALARM_COOLDOWN    = 3.0    # Seconds between repeated alarms

# ─── Global State ──────────────────────────────────────────────────────────────
alarm_active      = False
alarm_thread      = None
closed_frame_count = 0
last_alarm_time   = 0.0

# ─── Audio: Generate a buzzer beep using pygame ────────────────────────────────
def generate_beep(frequency=880, duration=0.4, volume=0.8, sample_rate=44100):
    """Generate a numpy sine-wave beep and return as pygame Sound."""
    if not PYGAME_AVAILABLE:
        return None
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
    # Make stereo
    stereo = np.column_stack([wave, wave])
    sound = pygame.sndarray.make_sound(stereo)
    return sound

def init_audio():
    """Initialize pygame mixer."""
    if not PYGAME_AVAILABLE:
        return None
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        return generate_beep()
    except Exception as e:
        print(f"[WARNING] Audio init failed: {e}")
        return None

# ─── Eye Aspect Ratio (EAR) ────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    Compute Eye Aspect Ratio from 6 landmark points.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    A low EAR (< threshold) indicates a closed eye.
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * img_w, lm.y * img_h]))

    # Vertical distances
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])

    if C == 0:
        return 0.3  # Avoid division by zero
    ear = (A + B) / (2.0 * C)
    return ear

# ─── Alarm Popup Window ────────────────────────────────────────────────────────
def show_alarm_window(beep_sound):
    """
    Display a full-screen red popup alarm with white text.
    Loops beep sound until dismissed.
    Runs in a separate thread.
    """
    global alarm_active

    # Create alarm window
    alarm_win = np.zeros((480, 640, 3), dtype=np.uint8)
    alarm_win[:] = (0, 0, 220)  # Red background (BGR)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    text1      = "DO NOT SLEEP!"
    text2      = "DRIVER ALERT!"
    text3      = "Open your eyes to dismiss"

    # Center text
    t1_size = cv2.getTextSize(text1, font, 2.0, 4)[0]
    t2_size = cv2.getTextSize(text2, font, 1.5, 3)[0]
    t3_size = cv2.getTextSize(text3, font, 0.7, 2)[0]

    t1_x = (640 - t1_size[0]) // 2
    t2_x = (640 - t2_size[0]) // 2
    t3_x = (640 - t3_size[0]) // 2

    cv2.putText(alarm_win, text1, (t1_x, 180), font, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(alarm_win, text2, (t2_x, 280), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(alarm_win, text3, (t3_x, 380), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw warning icon (triangle)
    pts_tri = np.array([[320, 60], [260, 140], [380, 140]], np.int32)
    cv2.polylines(alarm_win, [pts_tri], True, (255, 255, 255), 3)
    cv2.putText(alarm_win, "!", (311, 132), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.namedWindow("DROWSINESS ALERT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DROWSINESS ALERT", 640, 480)

    beep_interval = 0.5  # seconds between beeps
    last_beep     = 0.0

    while alarm_active:
        cv2.imshow("DROWSINESS ALERT", alarm_win)
        cv2.waitKey(30)

        # Play beep
        now = time.time()
        if beep_sound and now - last_beep > beep_interval:
            try:
                beep_sound.play()
            except Exception:
                pass
            last_beep = now

    # Alarm dismissed — close window
    try:
        cv2.destroyWindow("DROWSINESS ALERT")
    except Exception:
        pass

def trigger_alarm(beep_sound):
    """Start the alarm in a background thread (non-blocking)."""
    global alarm_active, alarm_thread, last_alarm_time

    now = time.time()
    if alarm_active:
        return  # Already ringing
    if now - last_alarm_time < ALARM_COOLDOWN:
        return  # Too soon to re-alarm

    alarm_active    = True
    last_alarm_time = now
    alarm_thread    = threading.Thread(target=show_alarm_window, args=(beep_sound,), daemon=True)
    alarm_thread.start()
    print("[ALARM] Drowsiness detected — alarm triggered!")

def dismiss_alarm():
    """Stop the alarm (called when eyes are open again)."""
    global alarm_active
    if alarm_active:
        alarm_active = False
        print("[INFO] Eyes open — alarm dismissed.")

# ─── Draw HUD on frame ─────────────────────────────────────────────────────────
def draw_hud(frame, ear_avg, closed_frame_count, alarm_active, face_detected):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)

    if not face_detected:
        cv2.putText(frame, "No face detected", (10, 45), font, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
        return frame

    # EAR value
    ear_color = (0, 255, 0) if ear_avg >= EAR_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, f"EAR: {ear_avg:.3f}", (10, 45), font, 1.0, ear_color, 2, cv2.LINE_AA)

    # Frame counter
    cv2.putText(frame, f"Closed: {closed_frame_count}/{CONSEC_FRAMES}", (210, 45), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    # Status
    if alarm_active:
        status_text  = "  DROWSY!"
        status_color = (0, 0, 255)
    elif closed_frame_count > 0:
        status_text  = "  Eyes Closing..."
        status_color = (0, 165, 255)
    else:
        status_text  = "  Awake"
        status_color = (0, 255, 0)

    cv2.putText(frame, status_text, (430, 45), font, 0.9, status_color, 2, cv2.LINE_AA)

    # EAR progress bar
    bar_x, bar_y, bar_w, bar_h = 10, 55, 200, 10
    fill = int(bar_w * min(ear_avg / 0.4, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), ear_color, -1)

    # Instructions at bottom
    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "Press Q to quit", (10, h - 8), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    return frame

# ─── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    global closed_frame_count, alarm_active

    print("=" * 55)
    print("  Driver Drowsiness Detection System")
    print("  Press Q in the camera window to quit")
    print("=" * 55)

    # Init audio
    beep_sound = init_audio()

    # Open webcam (0 = default/built-in)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check permissions or camera index.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # MediaPipe Face Mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces      = 1,
        refine_landmarks   = True,
        min_detection_confidence = 0.6,
        min_tracking_confidence  = 0.6
    )

    print("[INFO] Camera opened. Starting detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural feel
        h, w  = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        face_detected = False
        ear_avg       = 0.3  # Default "open"

        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark

            # Compute EAR for both eyes
            left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear_avg   = (left_ear + right_ear) / 2.0

            # Draw eye landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 200), -1)

            # Drowsiness logic
            if ear_avg < EAR_THRESHOLD:
                closed_frame_count += 1
                if closed_frame_count >= CONSEC_FRAMES:
                    trigger_alarm(beep_sound)
            else:
                # Eyes are open
                if closed_frame_count >= CONSEC_FRAMES:
                    dismiss_alarm()
                closed_frame_count = max(0, closed_frame_count - 2)  # Gradual decay

        else:
            # No face — reset counter but don't alarm on no-face
            closed_frame_count = max(0, closed_frame_count - 1)
            if closed_frame_count == 0:
                dismiss_alarm()

        # Draw HUD
        frame = draw_hud(frame, ear_avg, closed_frame_count, alarm_active, face_detected)

        cv2.imshow("Driver Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("[INFO] Quit key pressed. Exiting...")
            break

    # Cleanup
    alarm_active = False
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    if PYGAME_AVAILABLE:
        pygame.mixer.quit()
    print("[INFO] Program ended.")

if __name__ == "__main__":
    main()
