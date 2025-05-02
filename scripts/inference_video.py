import cv2
import numpy as np
import os
import joblib
import mediapipe as mp
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the trained model
loaded_model = load_model('models/Model_226_labels0.9608.h5')

# Load the label encoder and scaler
label_encoder = joblib.load('models/label_encoder0.9608.pkl')
scaler = joblib.load('models/scaler0.9608.pkl')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Helper function to extract holistic landmarks
def extract_holistic_landmarks(frame, holistic):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    hand_landmarks = []
    face_landmarks = []
    pose_landmarks = []

    if results.left_hand_landmarks:
        hand_landmarks.extend([(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        hand_landmarks.extend([(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark])
    if results.face_landmarks:
        face_landmarks.extend([(lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark])
    if results.pose_landmarks:
        pose_landmarks.extend([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])

    return {
        'hand_landmarks': np.array(hand_landmarks) if hand_landmarks else None,
        'face_landmarks': np.array(face_landmarks) if face_landmarks else None,
        'pose_landmarks': np.array(pose_landmarks) if pose_landmarks else None,
    }

# Process video frames and make predictions
def process_video(video_path):
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
    video_landmarks = []  

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        landmarks = extract_holistic_landmarks(frame, holistic)

        combined_landmarks = []
        if landmarks['hand_landmarks'] is not None:
            combined_landmarks.extend(landmarks['hand_landmarks'])
        if landmarks['face_landmarks'] is not None:
            combined_landmarks.extend(landmarks['face_landmarks'])
        if landmarks['pose_landmarks'] is not None:
            combined_landmarks.extend(landmarks['pose_landmarks'])

        if combined_landmarks:  
            video_landmarks.append(combined_landmarks)

    cap.release()
    holistic.close()

    x_array = np.array(video_landmarks, dtype=object)

    if x_array.size > 0:
        max_length = 543
        x_padded = pad_sequences(x_array, maxlen=max_length, padding='post', dtype='float32')
        flattened_data = x_padded.reshape(-1, x_padded.shape[-1])
        landmarks_data_scaled = scaler.transform(flattened_data)
        landmarks_data_scaled = landmarks_data_scaled.reshape(x_padded.shape)
        return landmarks_data_scaled
    else:
        return None  

# Evaluate and predict for video directory
def evaluate_video_directory(root_dir):
    overall_correct_predictions = 0
    overall_videos_count = 0
    per_adjective_results = {}

    for adjective in os.listdir(root_dir):
        adjective_path = os.path.join(root_dir, adjective)
        print(f"\nProcessing adjective: {adjective}")  
        total_correct_predictions = 0
        total_videos_count = 0

        if os.path.isdir(adjective_path):
            for video_file in os.listdir(adjective_path):
                video_path = os.path.join(adjective_path, video_file)
                print(f"Processing video: {video_file}") 
                landmarks = process_video(video_path)

                if landmarks is not None and len(landmarks) > 0:  
                    prediction = loaded_model.predict(landmarks)
                    predicted_classes = np.argmax(prediction, axis=1)
                    predicted_labels = label_encoder.inverse_transform(predicted_classes)

                    final_label, count = np.unique(predicted_labels, return_counts=True)
                    most_frequent_label = final_label[np.argmax(count)]
                    print(f"Video: {video_file} - Predicted label: {most_frequent_label}")       

                    if most_frequent_label == adjective:
                        total_correct_predictions += 1  
                    total_videos_count += 1
                else:
                    print(f"No valid landmarks found for video: {video_file}.")

        if total_videos_count > 0:
            accuracy = total_correct_predictions / total_videos_count
            print(f"Total Videos: {total_videos_count}, Correct Predictions: {total_correct_predictions}")
            print(f"Accuracy for adjective '{adjective}': {accuracy:.2%}")
            per_adjective_results[adjective] = {
                "correct": total_correct_predictions,
                "total": total_videos_count,
                "accuracy": accuracy
            }
            overall_correct_predictions += total_correct_predictions
            overall_videos_count += total_videos_count
        else:
            print(f"No videos found for adjective: {adjective}.")

    # Final summary
    print("\n--- Overall Summary ---")
    for adj, stats in per_adjective_results.items():
        print(f"{adj}: {stats['correct']}/{stats['total']} correct "
              f"({stats['accuracy']:.2%})")

    if overall_videos_count > 0:
        overall_accuracy = overall_correct_predictions / overall_videos_count
        print(f"\nTotal Videos: {overall_videos_count}, Total Correct Predictions: {overall_correct_predictions}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
    else:
        print("No videos were processed.")


# Main program
if __name__ == "__main__":
    root_dir = 'data/raw'  # Change to your video directory path
    evaluate_video_directory(root_dir)
