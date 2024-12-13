# Import required libraries
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import requests
import os
import io
from PIL import Image

# Telegram bot configuration
telegram_token = '6393874971:AAFotxMk31hfHedfM3Q1HXZR5tHhT3nNDrU'
chat_id = '1112715447'

# Function to send text messages to Telegram
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{telegram_token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, data=payload)
    return response

# Function to send photos to Telegram
def send_telegram_photo(photo, caption):
    url = f'https://api.telegram.org/bot{telegram_token}/sendPhoto'
    files = {'photo': photo}
    data = {'chat_id': chat_id, 'caption': caption}
    response = requests.post(url, files=files, data=data)
    return response

# Load YOLO model
model = YOLO('bestv2.pt')

# Function to get coordinates on mouse move
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

# Create window to display video
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Check current working directory
print("Current working directory:", os.getcwd())

# List directory contents to ensure files are present
print("Directory contents:", os.listdir())

# Open video file
cap = cv2.VideoCapture('cr.mp4')

# List of classes from your YOLO model
class_list = [
    'accident', 'bajaj', 'bus', 'car', 'motorcycle', 'people', 'truck'
]

count = 0
accident_reported = False  # Flag to check if accident has been reported

# Loop to process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (800, 800))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    accident_detected = False
    accident_coordinates = None

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        confidence = row[4]  # Extract confidence score
        d = int(row[5])
        c = class_list[d]

        # Draw bounding box and add confidence score
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cvzone.putTextRect(frame, f'{c} {confidence:.2f}', (x1, y1), 1, 1)  # Display class and confidence

        if c.lower() == 'accident':  # Adjust this condition based on your labels
            accident_detected = True
            accident_coordinates = (x1, y1, x2, y2)

    if accident_detected:
        if not accident_reported:
            # Save frame as an image
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()

            # Create stream for sending image
            photo = io.BytesIO(image_bytes)
            photo.name = 'accident.jpg'

            # Caption for photo, including coordinates
            caption = f"Accident detected at coordinates: {accident_coordinates}"
            
            # Send photo with caption to Telegram
            send_telegram_photo(photo, caption)
            
            accident_reported = True
    else:
        accident_reported = False

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
