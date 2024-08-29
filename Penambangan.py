import cv2
import mediapipe as mp
import os

# Inisialisasi MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk mengekstrak gambar dari video
def extract_frames(video_path, output_folder, frame_interval, video_prefix, output_resolution):
	# Membuka video file
	cap = cv2.VideoCapture(video_path)

	# Memeriksa apakah video berhasil dibuka
	if not cap.isOpened():
		print(f"Error: Tidak dapat membuka video {video_path}.")
		return

	# Membuat output folder jika belum ada
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	# Mendapatkan daftar file yang sudah ada untuk menghindari duplikasi
	existing_files = os.listdir(output_folder)
	existing_frame_numbers = [int(f.split('_frame_')[-1].split('.jpg')[0]) for f in existing_files if f.startswith(video_prefix)]

	# Menginisialisasi variabel frame count
	frame_count = 0

	# Loop melalui setiap frame dalam video
	while True:
		ret, frame = cap.read()

		# Jika frame tidak dapat dibaca, keluar dari loop
		if not ret or frame is None:
			break

		# Mengekstrak setiap frame ke-n sesuai interval yang ditentukan
		if frame_count % frame_interval == 0:
			frame_name = f"{video_prefix}_frame_{frame_count}.jpg"
			frame_path = os.path.join(output_folder, frame_name)

			# Cek apakah frame sudah ada sebelumnya
			if frame_name in existing_files:
				print(f"Skipping {frame_name}, already exists.")
			else:
				fh, fw, fc = frame.shape

				# Konversi frame ke grayscale
				gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

				with mp_face_detection.FaceDetection(min_detection_confidence=0.9) as face_detection:
					results = face_detection.process(frame)
					if results.detections:
						detection = results.detections[0]
						print(detection.location_data.relative_bounding_box)
						b = detection.location_data.relative_bounding_box
						x_ = int(b.xmin * fw) if b.xmin > 0 else 0
						y_ = int(b.ymin * fh) if b.ymin > 0 else 0
						_x = int((b.xmin + b.width) * fw) if b.xmin + b.width < 1 else fw
						_y = int((b.ymin + b.height) * fh) if b.ymin + b.height < 1 else fh
						gray_frame = gray_frame[y_:_y, x_:_x]


				# Resize frame ke resolusi yang diinginkan
				resized_frame = cv2.resize(gray_frame, output_resolution)

				# Menyimpan frame dengan format yang sesuai
				cv2.imwrite(frame_path, resized_frame)
				print(f"Saved {frame_path}")

		frame_count += 1

	# Melepaskan video capture
	cap.release()
	print(f"Selesai mengekstrak gambar dari {video_path}.")

# Fungsi untuk memproses banyak video dalam sebuah folder
def process_multiple_videos(video_folder, output_folder, frame_interval, output_resolution):
	# Membaca semua file video dalam folder
	video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

	for video_file in video_files:
		video_path = os.path.join(video_folder, video_file)
		video_prefix = os.path.splitext(video_file)[0]  # Menggunakan nama file sebagai prefix
		person_name = video_file.split('_')[0]  # Menggunakan nama orang sebagai prefix
		output_subfolder = os.path.join(output_folder, person_name)  # Buat subfolder untuk setiap video

		# Memanggil fungsi untuk mengekstrak gambar dari setiap video
		extract_frames(video_path, output_subfolder, frame_interval, video_prefix, output_resolution)