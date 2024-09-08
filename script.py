# This is a Google Colab File. Make Sure to run this file in Google Colab for no Errors

# This are required dependencies for installation 
# !pip install opencv-python
# !pip install numpy
# !pip install mediapipe
# !pip install matplotlib
# !pip install tensorflow

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Ensure TensorFlow uses GPU if available, otherwise use CPU and disable mixed precision
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Using GPU with mixed precision.")
else:
    tf.keras.mixed_precision.set_global_policy('float32')  # Use float32 on CPU
    print("GPU not available. Using CPU with float32.")

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to extract facial landmarks using MediaPipe
def extract_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            x = landmark.x * frame.shape[1]
            y = landmark.y * frame.shape[0]
            landmarks.append((x, y))
        return np.array(landmarks)
    return None

# Function to process a single video and extract landmark sequences
def process_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frames = []
    probas = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            sequence.append(landmarks.flatten())
            frames.append(frame)

        if len(sequence) >= max_frames:
            sequence_padded = pad_sequences([sequence], maxlen=max_frames, padding='post', truncating='post', dtype='float32')[0]
            sequence_padded = np.array(sequence_padded).reshape(1, max_frames, 468, 2, 1)  # Adjust shape for Conv3D
            probas.append(cnn_model.predict(sequence_padded)[0][0])

    cap.release()

    if len(sequence) > 0:
        sequence_padded = pad_sequences([sequence], maxlen=max_frames, padding='post', truncating='post', dtype='float32')[0]
        sequence_padded = np.array(sequence_padded).reshape(max_frames, 468, 2, 1)  # Shape: (max_frames, num_landmarks, 2, 1)
        return sequence_padded, frames, probas
    return None, None, None

# Define the 3D CNN model for deepfake detection
def create_3d_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))  # Adjusted the input size
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', dtype='float32'))  # 0: Real, 1: Deepfake

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to reconstruct the original face from landmarks by connecting lines between landmarks
def reconstruct_original_face(landmarks, frame):
    original_frame = frame.copy()

    # Define the connections for the face mesh based on MediaPipe's topology
    connections = [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10), # Outer face
        (151, 108), (108, 69), (69, 104), (104, 68), (68, 71), (71, 107), (107, 151), # Left eye
        (336, 296), (296, 334), (334, 293), (293, 300), (300, 383), (383, 374), (374, 380), (380, 252), # Right eye
        # Additional connections can be added for the nose, mouth, etc.
    ]

    # Iterate over connections and draw lines between landmarks
    for connection in connections:
        point1 = landmarks[connection[0]]
        point2 = landmarks[connection[1]]
        cv2.line(original_frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 2)

    return original_frame

# Function to detect deepfake, display frames with heat vision, reconstruct face, and plot deepfake probabilities
def detect_deepfake_and_display(video_path, cnn_model, max_frames=30):
    sequence, frames, probas = process_video(video_path, max_frames=max_frames)

    if sequence is not None:
        sequence = sequence.reshape(1, max_frames, 468, 2, 1)  # Adjust shape for Conv3D
        prediction = cnn_model.predict(sequence)
        deepfake = prediction > 0.5

        # Plot deepfake probability over time
        plt.figure(figsize=(12, 6))
        plt.plot(probas, label='Deepfake Probability')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        plt.xlabel('Frame')
        plt.ylabel('Probability')
        plt.title('Deepfake Probability Over Time')
        plt.legend()
        plt.show()

        # Display the first frame of the video with heat vision if deepfake
        if frames:
            frame = frames[0]
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

            # Original frame
            ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"Original Frame")
            ax1.axis("off")

            # Gender Classification - Placeholder for gender classification
            ax3.text(0.5, 0.5, f"Deepfake: {'Yes' if deepfake else 'No'}", fontsize=15, ha='center', va='center')
            ax3.axis("off")

            if deepfake:
                heat_vision = np.zeros_like(frame, dtype=np.uint8)
                landmarks = extract_landmarks(frame)
                if landmarks is not None:
                    for landmark in landmarks:
                        cv2.circle(heat_vision, tuple(int(i) for i in landmark), 4, (0, 0, 255), -1)
                heat_vision = cv2.addWeighted(frame, 0.3, heat_vision, 0.7, 0)
                ax2.imshow(cv2.cvtColor(heat_vision, cv2.COLOR_BGR2RGB))
                ax2.set_title(f"Heat Vision - Deepfake Detected")

                # Reconstruct and display original face with connected landmarks
                original_frame = reconstruct_original_face(landmarks, frame)
                ax3.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
                ax3.set_title(f"Reconstructed Face")

            else:
                ax2.imshow(np.zeros_like(frame))
                ax2.set_title(f"Heat Vision - Real Video")

            ax2.axis("off")
            plt.show()
    else:
        print("Not enough frames to make a prediction.")

# Example usage
input_shape = (30, 468, 2, 1)

##########
from google.colab import files

# Upload the video file
uploaded = files.upload()

# Get the path of the uploaded file
video_path = list(uploaded.keys())[0]
print(f"Uploaded video path: {video_path}")
##########

# Initialize your 3D CNN model
cnn_model = create_3d_cnn_model(input_shape)

# Run the deepfake detection on the uploaded video
detect_deepfake_and_display(video_path, cnn_model)
############
