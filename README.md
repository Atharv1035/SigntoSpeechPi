# **Indian Sign Language Recognition on Raspberry Pi 5**

### **Overview**
This project aims to bridge the communication gap between the mute and normal-speaking individuals by developing a product that can convert **Indian Sign Language (ISL) to text and speech in real-time**. It utilizes a **Raspberry Pi 5** along with an **RPi Camera Module v3** to capture hand gestures and translate them using a **Mediapipe + LSTM-based machine learning model**.

This project is developed as a part of the academic course: **"Electronics In Service to Society"**.

## **Features**
-  **Real-time ISL Recognition** – Converts sign language to text & speech instantly.
-  **Edge Computing on Raspberry Pi 5** – Runs locally without requiring cloud connectivity.
-  **Mediapipe Hand Tracking + LSTM Model** – Efficient gesture recognition.
-  **Portable and Cost-Effective** – Designed for practical, everyday use.

## **Hardware Requirements**
- **Raspberry Pi 5**
- **RPi Camera Module v3** (or any compatible USB camera)
- **MicroSD Card (32GB recommended)**
- **Power Adapter for Raspberry Pi 5**

## **Software Requirements**
- **Raspberry Pi OS (64-bit)**
- **Python 3.9+**
- **Mediapipe** (for hand tracking)
- **TensorFlow Lite** (for efficient LSTM model inference)
- **OpenCV** (for image processing)
- **Flite** (for text-to-speech conversion)

## **How It Works**
1. **Hand Gesture Detection:**
   - The **RPi Camera Module v3** captures hand movements in real-time.
   - **Mediapipe Hands** detects **21 hand landmarks**.

2. **Gesture Classification:**
   - A pre-trained **LSTM model** processes the landmark sequences.
   - The model identifies the corresponding sign from the dataset.

3. **Text & Speech Output:**
   - The recognized sign is displayed as **text on screen**.
   - **Flite TTS** converts text to speech output for audio feedback.

## **Dataset & Training**
- The model is trained on a **pre-existing dataset of ISL hand gestures word level**. ***[Dataset](https://www.kaggle.com/datasets/daskoushik/include)***
- Data collection done using **Mediapipe Hand Landmarks**.
- The process of landmark extraction can be found in [`process_videos.ipynb`](./process_videos.ipynb).
- Model is trained using *Bidirectional LSTM + Attention*. Training process can be found in [`train.ipynb`](./train.ipynb)
- The **Trained model** is stored in `models` directory.
- The converted **TFlite model** is stored in `rpi` directory.
- [`run_in_rpi.py`](./rpi/run_in_rpi.py) files runs inference on RPi5 for the model through Picamera v3 for realtime detection.
- LSTM model to be trained using **TensorFlow** and later converted to **TFLite** for optimized inference on Raspberry Pi.

## **Contributors**
- **Atharv Patil**
- **Atharva Kashalkar**
- **Saish Karole**
- **Atharva Atre**
- **Swar Kamatkar**



