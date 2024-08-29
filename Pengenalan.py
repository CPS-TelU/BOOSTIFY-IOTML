import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, RandomBrightness, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import penyimpanan as spn

def create_model(num_classes):
	model = Sequential([
		Input(shape=(96,96,1)),
		Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
		MaxPooling2D(pool_size=(2, 2)),
		Dropout(0.2),
		Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
		MaxPooling2D(pool_size=(2, 2)),
		Dropout(0.2),
		Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
		MaxPooling2D(pool_size=(2, 2)),
		Dropout(0.2),
		Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
		MaxPooling2D(pool_size=(2, 2)),
		Flatten(),
		Dense(512, activation='relu'),
		Dropout(0.2),
		Dense(num_classes, activation='softmax')
	])
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def visualize_history(training_recap):
	plt.figure(figsize=(12, 6))
	plt.subplot(1, 2, 1)
	plt.plot(training_recap.history['accuracy'])
	plt.plot(training_recap.history['val_accuracy'])
	plt.title('Model Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(['Train', 'Validation'])

	plt.subplot(1, 2, 2)
	plt.plot(training_recap.history['loss'])
	plt.plot(training_recap.history['val_loss'])
	plt.title('Model Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(['Train', 'Validation'])

# Function to visualize predictions
def visualize_predictions(model, data_gen, num_images=25):
	images, labels = next(data_gen)
	predictions = model.predict(images)
	class_labels = list(data_gen.class_indices.keys())
	
	for i in range(num_images):
		plt.subplot(1, num_images, i + 1)
		plt.figure(figsize=(6,6))
		plt.imshow(images[i])
		plt.title(f"True: {class_labels[np.argmax(labels[i])]}, Pred: {class_labels[np.argmax(predictions[i])]}") 
		plt.axis('off')
	plt.show()

def training(dataset, model_file):
	datagen = ImageDataGenerator(
		validation_split=0.3,
		rescale=1./255,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest'
	)

	# Load data
	train_data = datagen.flow_from_directory(
		dataset,
		target_size=spn.output_resolution,
		color_mode='grayscale',
		batch_size=32,
		class_mode='categorical',
		subset='training'
	)

	val_data = datagen.flow_from_directory(
		dataset,
		target_size=spn.output_resolution,
		color_mode='grayscale',
		batch_size=32,
		class_mode='categorical',
		subset='validation'
	)
	
	num_people = len(train_data.class_indices)
	print("number of people\t:", num_people)

	model_checkpoint = ModelCheckpoint(spn.checkpoint_file, save_best_only=True, monitor='val_loss')

	model_smile = create_model(num_people)
	model_smile.summary()

	# Melatih model
	history = model_smile.fit(train_data, epochs=2, batch_size=16, validation_data=val_data, callbacks=[model_checkpoint])

	model_smile.save(model_file)

	model_coba = load_model(model_file)

	# Define test data generator
	test_datagen = ImageDataGenerator(rescale=1./255)

	test_data = test_datagen.flow_from_directory(
		dataset,
		target_size=spn.output_resolution,
		color_mode='grayscale',
		batch_size=32,
		class_mode='categorical'
	)

	# Evaluate models
	print("Evaluating model for smiles:")
	model_coba.evaluate(test_data)

	visualize_history(history)
	visualize_predictions(model_coba, test_data)