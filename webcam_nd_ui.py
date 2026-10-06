from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import numpy as np
import gradio as gr

# Load the model
processor = ViTImageProcessor.from_pretrained('abhilash88/face-emotion-detection')
model = ViTForImageClassification.from_pretrained('abhilash88/face-emotion-detection')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Map emotions to background colors
emotion_colors = {
    'Angry': (255, 0, 0),       # Red
    'Disgust': (0, 128, 0),     # Dark Green
    'Fear': (128, 0, 128),      # Purple
    'Happy': (255, 255, 0),     # Yellow
    'Sad': (0, 0, 255),         # Blue
    'Surprise': (255, 165, 0),  # Orange
    'Neutral': (200, 200, 200)  # Gray
}

def predict_emotion(frame):
    # Convert webcam frame (numpy array) to PIL image
    image = Image.fromarray(frame.astype(np.uint8))

    # Predict emotion
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    
    emotion_label = emotions[pred]
    confidence = probs[0][pred].item()

    # Change background color based on emotion
    color = emotion_colors[emotion_label]
    overlay = np.full(frame.shape, color, dtype=np.uint8)
    blended = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    # Put text on the frame
    import cv2
    cv2.putText(blended, f"{emotion_label} ({confidence*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return blended

# Gradio webcam interface
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(source="webcam", tool="editor", type="numpy"),
    outputs="numpy",
    live=True,
    title="Webcam Emotion Detector",
    description="Your background changes based on detected emotion!"
)

iface.launch()
