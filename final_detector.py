import os
import warnings

# 1. Hide the "junk" text and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import cv2
import mediapipe as mp
import time

# 2. Setup MediaPipe (Standard Beginner Setup)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 3. Basic Variables
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 20
counter = 0
last_print_time = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image so it acts like a mirror
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # Convert color for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_ear = 0 # Starting value

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        
        # --- LEFT EYE CALCULATION (Step-by-Step) ---
        # Get coordinates for the left eye
        p362 = mesh_points[362] # Left side
        p263 = mesh_points[263] # Right side
        p385 = mesh_points[385] # Top
        p373 = mesh_points[373] # Bottom
        
        left_v = abs(p385.y - p373.y) # Vertical distance
        left_h = abs(p362.x - p263.x) # Horizontal distance
        left_ear = left_v / left_h

        # --- RIGHT EYE CALCULATION (Step-by-Step) ---
        p33 = mesh_points[33]   # Left side
        p133 = mesh_points[133] # Right side
        p160 = mesh_points[160] # Top
        p153 = mesh_points[153] # Bottom

        right_v = abs(p160.y - p153.y)
        right_h = abs(p33.x - p133.x)
        right_ear = right_v / right_h

        # Average of both eyes
        current_ear = (left_ear + right_ear) / 2.0

        # 4. Drowsiness Logic
        if current_ear < EAR_THRESHOLD:
            counter = counter + 1
            
            # If eyes stay closed long enough
            if counter >= CONSEC_FRAMES:
                # Print to terminal only once a second
                if time.time() - last_print_time > 1:
                    print(f"[{time.strftime('%H:%M:%S')}] STATUS: DROWSY")
                    last_print_time = time.time()

                # Visual Alert (Red Box)
                cv2.rectangle(frame, (0,0), (width, height), (0,0,255), 20)
                cv2.putText(frame, "!!! ALERT !!!", (width//2 - 100, height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
        else:
            counter = 0 # Reset if eyes open

    # 5. DRAW THE BAR (Simple Rectangles)
    # Draw gray background bar
    cv2.rectangle(frame, (50, 50), (300, 80), (50, 50, 50), -1)
    
    # Calculate how long the green/red bar should be
    bar_length = int(current_ear * 700)
    if bar_length > 250: bar_length = 250 # Cap the length
    
    # Pick color: Green if awake, Red if sleepy
    if current_ear < EAR_THRESHOLD:
        color = (0, 0, 255) # Red
    else:
        color = (0, 255, 0) # Green
        
    cv2.rectangle(frame, (50, 50), (50 + bar_length, 80), color, -1)
    cv2.putText(frame, "EAR VALUE", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the window
    cv2.imshow('Drowsiness Detector', frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
