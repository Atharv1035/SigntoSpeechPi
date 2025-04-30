import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import joblib
import tensorflow as tf
import mediapipe as mp
import cv2

# ==== Load Models (cached) ====
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler0.8953.pkl")
    label_encoder = joblib.load("models/label_encoder0.8953.pkl")
    interpreter = tf.lite.Interpreter(model_path="models/255_labels0.8953.tflite")
    interpreter.allocate_tensors()
    return scaler, label_encoder, interpreter

scaler, label_encoder, interpreter = load_models()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== Video Transformer for Streamlit-WeRTC ====
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize MediaPipe Holistic for this transformer instance
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1
        )
        self.prev_prediction = ""
        self.prev_confidence = 0.0

    def transform(self, frame):
        # Convert frame to numpy BGR image
        img = frame.to_ndarray(format="bgr24")

        # Process frame with MediaPipe
        results = self.holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Prepare drawing specifications
        drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img,
                results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        # Draw face landmarks
        if results.face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img,
                results.face_landmarks,
                mp.solutions.holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img,
                results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img,
                results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        # Extract and prepare all landmarks for inference
        all_lms = []
        for lm_list in [results.pose_landmarks, results.face_landmarks,
                        results.left_hand_landmarks, results.right_hand_landmarks]:
            if lm_list and hasattr(lm_list, 'landmark'):
                all_lms.extend([[lm.x, lm.y, lm.z] for lm in lm_list.landmark])

        landmarks_arr = np.array(all_lms)
        # Pad or truncate to exactly 543 landmarks
        if landmarks_arr.shape[0] < 543:
            landmarks_arr = np.pad(landmarks_arr, ((0, 543 - landmarks_arr.shape[0]), (0, 0)), mode='constant')
        elif landmarks_arr.shape[0] > 543:
            landmarks_arr = landmarks_arr[:543]

        # Run TFLite inference
        try:
            scaled = scaler.transform(landmarks_arr.reshape(-1, 3)).reshape(543, 3)
            input_tensor = np.expand_dims(scaled, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            idx = np.argmax(output)
            self.prev_prediction = label_encoder.inverse_transform([idx])[0]
            self.prev_confidence = float(np.max(output))
        except Exception:
            pass

        # Overlay prediction text on frame
        if self.prev_confidence > 0.8:
            cv2.putText(
                img,
                f"{self.prev_prediction} ({self.prev_confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
           
        return img

# ==== Streamlit App ==== 
st.title("🤟 Real-Time Sign Language Prediction")
st.write("Using your webcam and MediaPipe for continuous gesture recognition.")
webrtc_streamer(
    key="sign_language",
    video_transformer_factory=SignLanguageTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
