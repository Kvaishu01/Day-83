import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Detection App", page_icon="📸", layout="wide")

st.title("👁️ Face Detection using OpenCV + Haar Cascade")
st.write("Upload an image or use webcam to detect faces in real-time!")

# Load pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

option = st.radio("Choose Mode:", ["📸 Upload Image", "🎥 Webcam Detection"])

# --- Image Upload Mode ---
if option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

        st.image(img_array, caption=f"Detected {len(faces)} face(s)", use_column_width=True)

# --- Webcam Detection Mode ---
elif option == "🎥 Webcam Detection":
    st.warning("Press **Start** to begin real-time detection (Requires camera access).")

    start = st.button("Start Webcam")

    if start:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

            # Stop when 'Stop' button is clicked
            if st.button("Stop Webcam"):
                break

        cap.release()
        st.success("Webcam stopped.")

st.markdown("---")
st.markdown("**Developed as part of Day 83 – 100 Days of ML Challenge 🌟**")
