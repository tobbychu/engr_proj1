import cv2
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

import torchvision
from torchvision import datasets, models, transforms
from facenet_pytorch import InceptionResnetV1

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class FacialRecognition:
    def __init__(self):
        self.num_faces_recognized = 0

    def setup_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load pre-trained model.
        model = InceptionResnetV1(pretrained='vggface2', device=device, classify= True, num_classes=2)

        for param in model.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default.
        # num_features = model.fc.in_features

        # # We instantiate a new linear layer as our final classifier layer.
        # model.fc = nn.Linear(num_features, 2, bias=True)
        model = model.to(device)

        path = "models/trained_model.pt"
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

        model.eval()

        # Predict on detected images.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.Resize([224, 244]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return model, transform

    def recognize_face(self, model, transform, frame, faces_bounding_box):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        labels = ['angelina', 'not_angelina']
        label_color = [(255, 255, 255), (255, 255, 255)]
        self.num_faces_recognized = 0

        for face_bounding_box in faces_bounding_box:

            x1 = face_bounding_box[0]
            y1 = face_bounding_box[1]
            x2 = face_bounding_box[2]
            y2 = face_bounding_box[3]

            # Clamp coordinates that are outisde of the image.
            #x_min, y_min = max(x_min, 0), max(y_min, 0)
            x1, y1 = max(x1, 0), max(y1, 0)

            # Crop frame to detected face.
            detected_face = frame[y1:y2, x1: x2]
            pil_detected_face = Image.fromarray(detected_face)

            # Transform image (to tensor).
            image_transformed = transform(pil_detected_face)

            # Unsqueeze tensor (add 1 more dimension).
            inputs = image_transformed.unsqueeze(0)

            # We turn off gradients.
            with torch.no_grad():
                # Get prediction.
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                result = labels[preds]
    
                if result == "angelina":
                    print(f"adding to num_facial_recognition: {self.num_faces_recognized}")
                    self.num_faces_recognized += 1

                # Display prediction on frame.
                cv2.putText(frame, result, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame, result
        
    def print_num_faces_recognized(self, frame):
        # Print number of facial recognition.
        cv2.putText(frame, f"NUMBER OF FACES RECOGNIZED: {self.num_faces_recognized}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        return frame

if __name__ == "__main__":
   fr = FacialRecognition()
   model, transform = fr.setup_model()