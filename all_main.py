import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from math import atan2, degrees
import os
from datetime import datetime
import supabase
from database import db1, user1
import time
from audioplayer import play_random_audio

face_recognition_model = tf.keras.models.load_model('face_recognition.keras')
smile_detection_model = tf.keras.models.load_model('smile_detection_model.h5')

# Class names for face recognition
class_names = []

class_names = [f for f in os.listdir("DATA/known faces")] 

class_names.sort()
last = class_names[-1]
for i in range(len(class_names)-2, -1, -1):
    if last == class_names[i]:
        del class_names[i]
    else:
        last = class_names[i]
        
print(class_names)

confidence_threshold = 95.0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

output_dir = 'D://project cps/captured_smile'
os.makedirs(output_dir, exist_ok=True)

file_counters = {name: 1 for name in class_names}  # Initialize file counter

last_save_time = {name: 0 for name in class_names}
cooldown_period = 2 * 60 * 60  # 2 hours in seconds
last_submission_time = {}
audio_cooldown_period = 2 * 60 * 60 
last_audio_play_time = {name: 0 for name in class_names}

def get_next_filename(label):
    if label == "Unknown":
        label = "unknown"  # Handle the unknown label case
    dt = datetime.now()
    filename = os.path.join(output_dir, f'{label}_attendance_{dt.strftime("%Y-%m-%d-%H-%M-%S")}.jpg')
    return filename, dt

def upload_image(file_bytes, file_name):
    # Extract the filename from the full path
    file_name = os.path.basename(file_name)
    file_name = file_name.replace("\\", "/")
    
    # Unggah file ke storage dalam bucket tertentu
    try:
        response = db1.storage.from_('bucket_cps').upload(f'captured_images/{file_name}', file_bytes)
        print(response.__dict__)

        if response.status_code != 200:
            print('Upload error:', response['error'])
            return None
        
        # Parse the response as JSON if applicable
        response_json = response.json()
        print('File uploaded successfully')
        return response_json.get('path')  # Adjust according to actual response structure
    except Exception as e:
        print('An error occurred when uploading:', str(e))
        return None

def handle_capture(image, code):
    global last_save_time  # Access the global last_save_time dictionary
    current_time = time.time()

    # Check if the cooldown period has passed
    if current_time - last_save_time[code] >= cooldown_period:
        file_name, timestamp = get_next_filename(code)

        timestamp = datetime.now()
        file_name = f'{code}_attendance_{timestamp.strftime("%Y-%m-%d-%H-%M-%S")}.jpg'

        # Convert captured image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        image_path = upload_image(image_bytes, file_name)
        
        

        if image_path:
            save_capture_info(code, image_path, timestamp)
            last_save_time[code] = current_time  # Update last save time

        # Check if the cooldown period has passed for playing audio
        if current_time - last_audio_play_time[code] >= audio_cooldown_period:
            play_random_audio('audioplayer/components')
            last_audio_play_time[code] = current_time  # Update last audio play time
    else:
        print(f"Cooldown active for {code}. Image not saved.")

def save_capture_info(class_names, timestamp:datetime):
    try:
        assisstant_code, name = class_names.split('_')
    except ValueError:
        print(f"Error: class_names '{class_names}' is not in the correct format.")
        return
    
    global last_submission_time
    current_time = time.time()
    person_key = f"{assisstant_code}_{name}"
    
    assisstant_code, name = class_names.split('_')
    # Check if the cooldown period has passed
    if person_key not in last_submission_time or (current_time - last_submission_time[person_key] >= cooldown_period):
        assisstant_code, name = class_names.split('_')
        
        data = {
            'assisstant_code': assisstant_code,
            'name': name,
            'time': timestamp.isoformat()
        }
        try:
            response = db1.schema('public').table('Attendance').insert(data).execute()
            last_submission_time[person_key] = current_time  # Update last submission time
            print("Table Insert Response\t: ", response)
            
        except Exception as e:
            print(f"An error occurred when inserting into the table: {e}")
    else:
        print(f"Cooldown in effect for {name}. Data not sent.")
                
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    smile_angle = 0.0
    
    # Process face landmarks using MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Detect mouth landmarks
            left_mouth = np.array([int(face_landmarks.landmark[61].x * w), int(face_landmarks.landmark[61].y * h)])
            right_mouth = np.array([int(face_landmarks.landmark[291].x * w), int(face_landmarks.landmark[291].y * h)])
            top_mouth = np.array([int(face_landmarks.landmark[13].x * w), int(face_landmarks.landmark[13].y * h)])
            bottom_mouth = np.array([int(face_landmarks.landmark[14].x * w), int(face_landmarks.landmark[14].y * h)])

            # Draw the mouth landmarks
            cv2.circle(frame, tuple(left_mouth), 2, (0, 255, 0), -1)
            cv2.circle(frame, tuple(right_mouth), 2, (0, 255, 0), -1)
            cv2.circle(frame, tuple(top_mouth), 2, (0, 255, 0), -1)
            cv2.circle(frame, tuple(bottom_mouth), 2, (0, 255, 0), -1)

            cv2.line(frame, tuple(left_mouth), tuple(right_mouth), (0, 255, 0), 1)
            cv2.line(frame, tuple(top_mouth), tuple(bottom_mouth), (0, 255, 0), 1)

            # Calculate smile angle
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            mouth_height = np.linalg.norm(top_mouth - bottom_mouth)
            smile_angle = degrees(atan2(mouth_height, mouth_width))

    for (x, y, w, h) in faces:
        # Extract face region of interest
        face_roi = gray_frame[y:y + h, x:x + w]

        resized_face = cv2.resize(face_roi, (96, 96))
        input_image = resized_face.reshape(1, 96, 96, 1) / 255.0

        predictions = face_recognition_model.predict(input_image)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        label = "Unknown" if confidence < confidence_threshold else class_names[predicted_class]

        smile_input_image = cv2.resize(face_roi, (28, 28))
        smile_input_image = smile_input_image.reshape(1, 28, 28, 1) / 255.0

        smile_prediction = smile_detection_model.predict(smile_input_image)
        smile_label = 'Smiling' if smile_prediction[0][1] > 0.5 else 'Smile more!!'

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label.split('_')[0]} ({confidence:.2f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"{smile_label} ({(smile_prediction[0][1] * 100):.2f}%)", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
        # Save image if smile_angle > 8 and person is smiling
        if smile_label == 'Smiling' and smile_angle > 6:
            filename, timestamp = get_next_filename(label)
            if label.lower() != "unknown":    
                face_image = frame[y:y + h, x:x + w]
                cv2.imwrite(filename, face_image)
                print(f"BOOSTIFY!!!!!, Image saved as '{filename}'")

            # Convert the image to bytes for uploading
                _, buffer = cv2.imencode('.jpg', face_image)
                image_bytes = buffer.tobytes()

            # Upload the image and handle capture (save information)
                handle_capture(face_image, label)
                save_capture_info(label, timestamp)
                # play_random_audio('audioplayer/components')
                
    cv2.putText(frame, f"{datetime.now()}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow('BOOSTIFY', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

db1.auth.sign_out()
db1.auth.close()
cap.release()
cv2.destroyAllWindows()
