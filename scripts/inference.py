import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from collections import deque
import time

# ==== Load Scaler & Label Encoder ====
scaler = joblib.load("/home/saishk/SigntoSpeechPi/models/scaler0.9475.pkl")
label_encoder = joblib.load("/home/saishk/SigntoSpeechPi/models/label_encoder0.9475.pkl")
print("Loaded labels:", label_encoder.classes_)

# ==== Load TFLite Model ====
interpreter = tf.lite.Interpreter(model_path="/home/saishk/SigntoSpeechPi/models/226_labels0.9475.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== Setup MediaPipe Holistic ====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==== Extract full 543 body landmarks with padding/truncation ====
def extract_full_body_landmarks(results):
    all_landmarks = []
    for lm_list in [
        results.left_hand_landmarks,
        results.right_hand_landmarks,
        results.face_landmarks,
        results.pose_landmarks
    ]:
        if lm_list:
            all_landmarks.extend([[lm.x, lm.y, lm.z] for lm in lm_list.landmark])

    landmark_array = np.array(all_landmarks)

    # Ensure shape is exactly (543, 3)
    if landmark_array.shape[0] < 543:
        landmark_array = np.pad(landmark_array, ((0, 543 - landmark_array.shape[0]), (0, 0)), mode='constant')
    elif landmark_array.shape[0] > 543:
        landmark_array = landmark_array[:543]

    return landmark_array

# ==== Webcam and Inference Setup ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

sequence = deque(maxlen=30)  # Increased sequence length for better context
predictions = deque(maxlen=30)

# Track frame rate (FPS)
prev_time = time.time()

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed.")
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Debugging: Draw bounding boxes around hands, face, and pose landmarks
        if results.left_hand_landmarks:
            hand_landmarks = results.left_hand_landmarks
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            cv2.rectangle(image, (int(min_x * image.shape[1]), int(min_y * image.shape[0])),
                          (int(max_x * image.shape[1]), int(max_y * image.shape[0])), (0, 255, 0), 2)

        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            cv2.rectangle(image, (int(min_x * image.shape[1]), int(min_y * image.shape[0])),
                          (int(max_x * image.shape[1]), int(max_y * image.shape[0])), (0, 255, 0), 2)

        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks
            x_vals = [lm.x for lm in pose_landmarks.landmark]
            y_vals = [lm.y for lm in pose_landmarks.landmark]
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            cv2.rectangle(image, (int(min_x * image.shape[1]), int(min_y * image.shape[0])),
                          (int(max_x * image.shape[1]), int(max_y * image.shape[0])), (0, 0, 255), 2)

        try:
            landmarks = extract_full_body_landmarks(results)  # Shape: (543, 3)

            if landmarks.shape == (543, 3):
                # Flatten to (543, 3) → Scaler expects this because training used (N, 3)
                landmarks_scaled = scaler.transform(landmarks)  # shape: (543, 3)
                sequence.append(landmarks_scaled)

                # Prepare input: (1, 543, 3)
                input_data = np.expand_dims(sequence[0], axis=0).astype(np.float32)

                # Run inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]

                predicted_label = np.argmax(output)
                prediction = label_encoder.inverse_transform([predicted_label])[0]

                # Confidence threshold for prediction
                if np.max(output) > 0.8:
                    print(f"[PREDICT] {prediction} | Confidence: {np.max(output):.2f}")
                    cv2.putText(image, f'{prediction}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        except Exception as e:
            print("Error during inference:", str(e))
            continue

        # Calculate FPS
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time

        # Display FPS
        cv2.putText(image, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Show the image
        cv2.imshow("Gesture Recognition", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
