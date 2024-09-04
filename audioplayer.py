import pygame
import os
import random

def play_random_audio(directory):
    # List all audio files in the specified directory
    audio_files = [f for f in os.listdir(directory) if f.endswith('.mp3') or f.endswith('.wav')]
    
    # Check if there are any audio files in the directory
    if not audio_files:
        print("No audio files found in the directory.")
        return
    
    # Randomly select an audio file
    selected_audio = random.choice(audio_files)
    file_path = os.path.join(directory, selected_audio)
    
    # Initialize the mixer
    pygame.mixer.init()
    
    # Load the randomly selected audio file
    pygame.mixer.music.load(file_path)
    
    # Play the audio file
    pygame.mixer.music.play()
    
    print(f"Now playing: {selected_audio}")
    
    # Keep the program running until the audio is done playing
    while pygame.mixer.music.get_busy():
        continue

# Example usage:
audio_directory = "/Magang cps/audio_player/components"  # Replace with the path to your audio files
play_random_audio(audio_directory)


# Buat mengintegrasi ke face detect
# def on_face_detected():
#     play_random_audio(audio_directory)

# if face_detected:  
#     on_face_detected()
