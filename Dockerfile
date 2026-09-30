FROM python:3.12-slim
WORKDIR /app
COPY step2_face_recognition.py .
CMD ["python", "-u", "step2_face_recognition.py"]