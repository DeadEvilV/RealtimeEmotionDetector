# Real-Time Emotion Detector

This is a project I worked on to learn about real-time emotion detection and understand how datasets work, along with how to create the architecture of an AI model. The goal was to build an emotion recognition system from scratch, focusing on the process of constructing and training the AI rather than achieving high accuracy. The current model has an accuracy of around 50%, which reflects the decision to not use pre-trained models. While these could improve performance, they would take away the learning experience.

The Convolutional Neural Network (CNN) I developed can recognize emotions such as happiness, sadness, surprise, anger, neutrality, and disgust from live video streams. By not relying on pre-trained AI, I was able to dive into understanding how to design and train these types of models.

Here’s how it works:

1. **Real-Time Video Processing**: The program uses OpenCV to process video frames in real time and detect faces.
2. **Custom Neural Network**: The CNN analyzes the detected face to classify the emotion based on the features it has learned during training.
3. **Live Feedback**: Emotions are displayed in real-time as the model processes the video feed, together with a confidence score of the prediction.

The purpose of this project was educational, so the model’s accuracy isn’t perfect.

## How to use
```bash
git clone https://github.com/DeadEvilV/RealtimeEmotionDetector.git
cd RealtimeEmotionDetector
pip install -r requirements.txt
python src/realtime_emotion_detector.py
```

If your computer does not have access to a webcam, you can download apps such as Iriun Webcam.

![image](https://github.com/user-attachments/assets/d57233e7-8e26-4159-a719-82b4a266c824)
![image](https://github.com/user-attachments/assets/896db1f4-67fc-4cfe-8b23-328b59f5ba5f)
