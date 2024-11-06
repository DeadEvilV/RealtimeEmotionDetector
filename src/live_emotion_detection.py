import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import EmotionDetecterCNN

def preprocess():
    mean = 0.5073
    std = 0.2120
    image_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((44, 44)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return image_transforms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionDetecterCNN(num_classes=7)
    checkpoint = torch.load('EmotionClassifierCNN.ckpt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear',
                           3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    confidence_threshold = 0.2
    image_transforms = preprocess()
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        if not ret:
            print('Failed to capture image')
            break
        
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            # crop to 44x44
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue # skip empty face
            
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            input = image_transforms(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input)
                probablities = torch.softmax(logits, dim=1)
                confidence, prediction = torch.max(probablities, dim=1)
                confidence = confidence.item()
                emotion = emotion_labels[prediction.item()]
            if confidence > confidence_threshold:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label_position = (x, y-10)
                label = f'{emotion} {confidence:.2f}'
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        cv2.imshow('Real Time Emotion Detection', frame)
        
        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
