FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir opencv-python-headless numpy paho-mqtt
COPY models/ models/
COPY step2_face_recognition.py .
CMD ["python", "-u", "step2_face_recognition.py"]