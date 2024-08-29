import os
import cv2
import mediapipe as mp
import numpy as np 
import tensorflow as tf
import Penambangan as mine
import penyimpanan as spn
from Pengenalan import training
from tensorflow.keras.utils import img_to_array
#from Pengenalan_denganekspresi import detect_faces, training

# Inisialisasi MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk mendeteksi wajah
def detect_faces(image):
	with mp_face_detection.FaceDetection(min_detection_confidence=0.9) as face_detection:
		results = face_detection.process(image)
		if results.detections:
			for detection in results.detections:
				mp_drawing.draw_detection(image, detection)
				print("wajah ditambahkan")
	return image

def predict_identity_and_expression(image):
	# Preproses gambar
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	image = cv2.resize(image, spn.output_resolution)
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# Prediksi identitas dan ekspresi
	preds = model.predict(image)

	# Mendapatkan label identitas dan ekspresi
	person = spn.people[np.argmax(preds)]

	return person, np.argmax(preds)

print(os.stat(spn.video_folder))
print(spn.people)

if os.path.exists(spn.model_file) == False:
	print("Melatih atau memperbarui model")
	video_names = []
	for name in os.listdir(spn.video_folder):
		video_names.append(name.split('_')[0])

	print(video_names)

	if os.path.exists(spn.output_folder) == False:
		print("Mengekstraksi file video masukan")
		# Memanggil fungsi untuk memproses banyak video
		mine.process_multiple_videos(spn.video_folder, spn.output_folder, spn.frame_interval, spn.output_resolution)
	
	print(os.stat(spn.output_folder))
	# Daftar identitas orang dan ekspresi
	# Mendapatkan daftar nama dan ada di dalam folder dataset
	existing_files = os.listdir(spn.output_folder)
	spn.people = [f.split('_')[0] for f in existing_files]  # Sesuaikan dengan identitas yang ada

	# Menghilangkan duplikasi pada nama
	spn.people.sort()
	last = spn.people[-1]
	for i in range(len(spn.people)-2, -1, -1):
		if last == spn.people[i]:
			del spn.people[i]
		else:
			last = spn.people[i]
		
	#pknl.load_face_encodings_from_subfolders(spn.output_folder)
	#spn.save_encodings(spn.encode_file)
	training(spn.output_folder, spn.model_file)

print(os.stat(spn.model_file))

model = tf.keras.models.load_model(spn.model_file)

# Menggunakan kamera untuk mendeteksi wajah dan ekspresi
cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	if not ret:
		break

	# Deteksi wajah
	face_frame = detect_faces(frame)
	
	# Prediksi identitas dan ekspresi
	person, p_pred = predict_identity_and_expression(face_frame)
	
	# Tampilkan identitas dan ekspresi yang terdeteksi
	cv2.putText(frame, f'{person}: {p_pred}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cv2.imshow('Face Identity & Expression Recognition', frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

