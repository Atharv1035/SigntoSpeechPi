import cv2
import joblib
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# Load model, label encoder, scaler
loaded_model = load_model('../models/Model_56_labels0.8743.h5')
label_encoder = joblib.load('../models/label_encoder0.8743.pkl')
scaler = joblib.load('../models/scaler0.8743.pkl')

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Helper: Extract landmarks
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

    return results, {
        'hand_landmarks': np.array(hand_landmarks) if hand_landmarks else None,
        'face_landmarks': np.array(face_landmarks) if face_landmarks else None,
        'pose_landmarks': np.array(pose_landmarks) if pose_landmarks else None,
    }

# Settings
predictions_queue = deque(maxlen=5)
frame_count = 0
PREDICT_EVERY_N_FRAMES = 2
CONFIDENCE_THRESHOLD = 0.4

# Webcam
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FPS, 30)
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        #frame = cv2.flip(frame, 1)
        frame_count += 1

        results, landmarks = extract_holistic_landmarks(frame, holistic)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        combined_landmarks = []
        if landmarks['hand_landmarks'] is not None:
            combined_landmarks.extend(landmarks['hand_landmarks'])
        if landmarks['face_landmarks'] is not None:
            combined_landmarks.extend(landmarks['face_landmarks'])
        if landmarks['pose_landmarks'] is not None:
            combined_landmarks.extend(landmarks['pose_landmarks'])

        if len(combined_landmarks) == 543 and frame_count % PREDICT_EVERY_N_FRAMES == 0:
            frame_landmarks = np.array(combined_landmarks).reshape(543, 3)

            frame_landmarks_flat = frame_landmarks.reshape(-1, 3)
            frame_landmarks_scaled = scaler.transform(frame_landmarks_flat)
            frame_landmarks_scaled = frame_landmarks_scaled.reshape(1, 543, 3)

            prediction = loaded_model.predict(frame_landmarks_scaled, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            predicted_confidence = float(np.max(prediction))

            predictions_queue.append((predicted_label, predicted_confidence))

        if predictions_queue:
            labels, confidences = zip(*predictions_queue)
            most_common_label = max(set(labels), key=labels.count)
            avg_confidence = np.mean([conf for lab, conf in zip(labels, confidences) if lab == most_common_label])

            if avg_confidence > CONFIDENCE_THRESHOLD:
                text = f"{most_common_label} ({avg_confidence*100:.2f}%)"
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # Show
        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
