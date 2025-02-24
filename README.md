# ESS-Project
# **Indian Sign Language Recognition on Raspberry Pi 5**

## **Overview**
This project aims to bridge the communication gap between the mute and normal-speaking individuals by developing a product that can convert **Indian Sign Language (ISL) to text and speech in real-time**. It utilizes a **Raspberry Pi 5** along with an **RPi Camera Module v3** to capture hand gestures and translate them using a **Mediapipe + LSTM-based machine learning model**.

This project is developed as a part of the academic course: **"Electronics In Service to Society"**.

---
## **Features**
✅ **Real-time ISL Recognition** – Converts sign language to text & speech instantly.
✅ **Edge Computing on Raspberry Pi 5** – Runs locally without requiring cloud connectivity.
✅ **Mediapipe Hand Tracking + LSTM Model** – Efficient gesture recognition.
✅ **Portable and Cost-Effective** – Designed for practical, everyday use.

---
## **Hardware Requirements**
- **Raspberry Pi 5**
- **RPi Camera Module v3** (or any compatible USB camera)
- **MicroSD Card (32GB recommended)**
- **Power Adapter for Raspberry Pi 5**
- **Speakers (for speech output)**

---
## **Software Requirements**
- **Raspberry Pi OS (64-bit)**
- **Python 3.9+**
- **Mediapipe** (for hand tracking)
- **TensorFlow Lite** (for efficient LSTM model inference)
- **OpenCV** (for image processing)
- **Flite** (for text-to-speech conversion)

---
## **Installation & Setup**
### **1. Set up Raspberry Pi 5**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y
```

### **2. Create and Activate a Virtual Environment**
```bash
python3 -m venv isl_env
source isl_env/bin/activate
```

### **3. Install Dependencies**
```bash
pip install numpy opencv-python mediapipe tensorflow tflite-runtime
pip install pyttsx3
```

### **4. Run the Sign Language Recognition Script**
```bash
python isl_recognition.py
```

---
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

---
## **Dataset & Training**
- The model is trained on a **custom dataset of ISL hand gestures**.
- Data collection was done using **Mediapipe Hand Landmarks**.
- LSTM model was trained using **TensorFlow** and later converted to **TFLite** for optimized inference on Raspberry Pi.

---
## **Contributors**
- **Atharv Patil**
- **Atharva Kashalkar**
- **Saish Karole**
- **Atharva Atre**
- **Swar Kamatkar**

---
## **Future Improvements**
- Add **dynamic sentence formation** from continuous gestures.
- Implement **multi-language support** for speech output.
- Optimize for **lower latency and better accuracy**.

---
## **License**
This project is open-source and available under the **MIT License**.


