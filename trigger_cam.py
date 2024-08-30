import cv2 as cv
import mediapipe as mp
import numpy as np
from math import atan2, degrees


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            left_mouth = np.array([int(face_landmarks.landmark[61].x * w), int(face_landmarks.landmark[61].y * h)])
            right_mouth = np.array([int(face_landmarks.landmark[291].x * w), int(face_landmarks.landmark[291].y * h)])
            top_mouth = np.array([int(face_landmarks.landmark[13].x * w), int(face_landmarks.landmark[13].y * h)])
            bottom_mouth = np.array([int(face_landmarks.landmark[14].x * w), int(face_landmarks.landmark[14].y * h)])

            cv.circle(frame, tuple(left_mouth), 2, (0, 255, 0), -1)
            cv.circle(frame, tuple(right_mouth), 2, (0, 255, 0), -1)
            cv.circle(frame, tuple(top_mouth), 2, (0, 255, 0), -1)
            cv.circle(frame, tuple(bottom_mouth), 2, (0, 255, 0), -1)

            cv.line(frame, tuple(left_mouth), tuple(right_mouth), (0, 255, 0), 1)
            cv.line(frame, tuple(top_mouth), tuple(bottom_mouth), (0, 255, 0), 1)

            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            mouth_height = np.linalg.norm(top_mouth - bottom_mouth)
            smile_angle = degrees(atan2(mouth_height, mouth_width))
            
            if smile_angle > 8:
                cv.imwrite('attendance.jpg', frame)
                print("BOOSTYFY!!!!!,Image saved as 'attendance.jpg'")
    
    cv.imshow('Boostify', frame)
    
    if cv.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
