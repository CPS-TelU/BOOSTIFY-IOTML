import os

# Path ke model
#model_file = 'D:/Downloads/expression_recognition_model.keras'
model_file = 'D:/Downloads/Source/recognition_model.keras'
checkpoint_file = 'D:/Downloads/Source/best_model.keras'

# Path ke folder video
video_folder = "D:/Downloads/Source/data raw"

# Folder output untuk menyimpan gambar yang diekstrak
output_folder = "D:/Downloads/Source/data_extract"

# Interval frame, misalnya setiap 30 frame
frame_interval = 10

# Resolusi output yang diinginkan (width, height)
output_resolution = (96, 96)

# Daftar identitas orang dan ekspresi
# Mendapatkan daftar nama yang ada di dalam folder dataset

people = []

if os.path.exists(output_folder):
	existing_files = os.listdir(output_folder)
	people = [f.split('_')[0] for f in existing_files]  # Sesuaikan dengan identitas yang ada

	# Menghilangkan duplikasi pada nama
	if people:
		people.sort()
		last = people[-1]
		for i in range(len(people)-2, -1, -1):
			if last == people[i]:
				del people[i]
			else:
				last = people[i]

expressions = ['senyum', 'tidak_senyum']  # Daftar ekspresi yang dikenali
