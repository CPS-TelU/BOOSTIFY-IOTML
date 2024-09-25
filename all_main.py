import cv2
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
import mediapipe as mp
from math import atan2, degrees
import os
from datetime import datetime
import supabase
from database import db1
import time
from audioplayer import play_random_audio
from ast import literal_eval

face_recognition_model = tf.keras.models.load_model('/home/cps/BOOSTIFY-IOTML/face_recognition.keras')
smile_detection_model = tf.keras.models.load_model('/home/cps/BOOSTIFY-IOTML/smile_detection_model.h5')
str_class_names = ""

# Class names for face recognition
try:
    imported_names = db1.storage.from_('bucket_cps').download('model_label/known_faces.txt')
    
    # Updates the names in the file
    if imported_names:
        file = open('/home/cps/BOOSTIFY_cache/known_faces.txt', 'w')
        str_class_names = str(imported_names)[2:-1]
        print(str_class_names)
        file.write(str_class_names)
        print('Face names downloaded successfully')
        
    
except Exception as e:
    print('An error occurred when downloading:', str(e))
    file = open('/home/cps/BOOSTIFY_cache/known_faces.txt', 'r')
    str_class_names = file.read()

class_names = literal_eval(str_class_names)
class_names.sort()
last = class_names[-1]
for i in range(len(class_names)-2, -1, -1):
    if last == class_names[i]:
        del class_names[i]
    else:
        last = class_names[i]
        
print(class_names)

confidence_threshold = 0.99

# Inisialisasi MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

output_dir = '/home/cps/BOOSTIFY_cache/captured_smile'
audio_dir = '/home/cps/BOOSTIFY_audio/components'
last_submission_dir = '/home/cps/BOOSTIFY_cache/last_submission.txt'
last_save_dir = '/home/cps/BOOSTIFY_cache/last_save.txt'
os.makedirs(output_dir, exist_ok=True)

cooldown_period = 1  # 1 day
last_submission_time = {}
last_save_time = {}

if os.path.exists(last_save_dir):
    with open(last_save_dir, 'r') as f:
        last_save_time = literal_eval(f.read())

if os.path.exists(last_submission_dir):
    with open(last_submission_dir, 'r') as f:
        last_submission_time = literal_eval(f.read())

else:
    try:
        # Mengambil semua entri dari tabel Attendance
        response = db1.table('Attendance').select('*').execute()
        entries = response.data

        # Memproses entri untuk mendapatkan yang terbaru per asisten
        for entry in entries:
            print(entry)
            assisstant_code = entry['assisstant_code']
            name = entry['name']
            time = entry['time']
            person_key = f"{assisstant_code}_{name}"
            
            # Memeriksa apakah entri saat ini lebih baru dari entri terakhir yang sudah ada
            if person_key not in last_submission_time or datetime.fromisoformat(time) > last_submission_time[person_key]:
                last_submission_time[person_key] = time

        with open(last_submission_dir, 'w') as f:
            f.write(str(last_submission_time))
            print(last_submission_time)

    except Exception as e:
        print(f"An error occured when fetching data: {e}")

# Check if the audio device is connected by playing audio
play_random_audio(audio_dir)    

# Fungsi untuk mendeteksi wajah
def detect_faces(image):
	fh, fw, fc = image.shape

	with mp_face_detection.FaceDetection(min_detection_confidence=0.9) as face_detection:
		results = face_detection.process(image)
		if results.detections:
			detection = results.detections[0]
			print(detection.location_data.relative_bounding_box)
			b = detection.location_data.relative_bounding_box
			x_ = int((b.xmin) * fw) if b.xmin > 0 else 0
			y_ = int((b.ymin) * fh) if b.ymin > 0 else 0
			_x = int((b.xmin + b.width) * fw) if b.xmin + b.width < 1 else fw
			_y = int((b.ymin + b.height) * fh) if b.ymin + b.height < 1 else fh
			return x_, y_, _x, _y
		else: return None

def predict_identity_and_expression(image, resolution, model):
	# Preproses gambar
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	image = cv2.resize(image, resolution)
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# Prediksi identitas dan ekspresi
	preds = model.predict(image)

	# Mendapatkan label identitas dan ekspresi
	conf = np.max(preds)
	class_num = np.argmax(preds)

	print(preds)
	return class_num, conf

def get_next_filename(label):
    if label == "Unknown":
        label = "unknown"  # Handle the unknown label case
    current_time = datetime.now()
    filename = os.path.join(output_dir, f'{label}_attendance_{current_time.strftime("%Y-%m-%d-%H-%M-%S")}.jpg')
    return filename, current_time

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
    global last_submission_time  # Access the global last_save_time dictionary
    timestamp = datetime.now()

    # Check if the cooldown period has passed
    checking = last_submission_time == {} or code not in last_submission_time or (timestamp - datetime.fromisoformat(last_submission_time[code])).days >= cooldown_period

    if checking:
        file_name, timestamp = get_next_filename(code)

        file_name = f'{code}_attendance_{timestamp.strftime("%Y-%m-%d-%H-%M-%S")}.jpg'

        # Convert captured image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        image_path = upload_image(image_bytes, file_name)
        
        if image_path:
            save_capture_info(code, image_path, timestamp)
            
            last_submission_time[code] = timestamp  # Update last save time

        
    
    else:
        print(f"Cooldown on saving active for {code}. Image not saved.")
    
    print(last_submission_time)
    with open(last_submission_dir, 'w') as f:
        f.write(str(last_submission_time))

def save_capture_info(class_names, timestamp:datetime):
    try:
        assisstant_code, name = class_names.split('_')
    except ValueError:
        print(f"Error: class_names '{class_names}' is not in the correct format.")
        return
    
    global last_save_time
    person_key = f"{assisstant_code}_{name}"
    
    assisstant_code, name = class_names.split('_')
    # Check if the cooldown period has passed
    if last_save_time == {} or person_key not in last_save_time or (timestamp - datetime.fromisoformat(last_save_time[person_key])).days >= cooldown_period:
        assisstant_code, name = class_names.split('_')
        
        data = {
            'assisstant_code': assisstant_code,
            'name': name,
            'time': timestamp.isoformat()
        }      
        
        try:
            response = db1.schema('public').table('Attendance').insert(data).execute()    
            print("Table Insert Response\t: ", response)
            
            last_save_time[person_key] = timestamp.isoformat()  # Update last submission time
            
        except Exception as e:
            print(f"An error occurred when inserting into the table: {e}")
        
    else:
        print(f"Cooldown on upload active for {name}. Data not sent.")

    with open(last_save_dir, 'w') as f:
        f.write(str(last_save_time))
                
while True:
	ret, frame = cap.read()
	if not ret:
		print("Error: Failed to capture image.")
		break
	
	coords = detect_faces(frame)
	if coords != None:
		x_start, y_start, x_end, y_end = coords
		if x_end * y_end < frame.size:
			face_frame = frame[y_start: y_end, x_start:x_end]

			predicted_class, confidence = predict_identity_and_expression(face_frame, (96, 96), face_recognition_model)
			label = "Unknown" if confidence < confidence_threshold else class_names[predicted_class]

			is_smiling, smile_confidence = predict_identity_and_expression(face_frame, (28, 28), smile_detection_model)
			smile_label = 'Smiling' if ((is_smiling == True) and (smile_confidence > 0.7)) else 'Smile more!!'

			cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
			cv2.putText(frame, f"{label.split('_')[0]} ({confidence:.2f})", (x_start, y_start - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
			cv2.putText(frame, f"{smile_label}", (x_start, y_end + 20),
						cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

			
			# Save image if person is smiling
			if smile_label == 'Smiling':
				filename, timestamp = get_next_filename(label)
				if label.lower() != "unknown":    
					face_image = frame[y_start:y_end, x_start:x_end]
					cv2.imwrite(filename, face_image)
					print(f"BOOSTIFY!!!!!, Image saved as '{filename}'")

				# Convert the image to bytes for uploading
					_, buffer = cv2.imencode('.jpg', face_image)
					image_bytes = buffer.tobytes()

				# Upload the image and handle capture (save information)
					handle_capture(face_image, label)
					save_capture_info(label, timestamp)
					
					play_random_audio(audio_dir)
				
	cv2.putText(frame, f"{datetime.now()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
	cv2.namedWindow('BOOSTIFY', cv2.WINDOW_NORMAL)
	cv2.imshow('BOOSTIFY', frame)
	cv2.resizeWindow('BOOSTIFY', 480, 320)
	#cv2.setWindowProperty('BOOSTIFY', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

db1.auth.sign_out()
db1.auth.close()
cap.release()
cv2.destroyAllWindows()
