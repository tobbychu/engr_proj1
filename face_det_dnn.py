import cv2
import numpy as np

class FaceDetectionDNN:
    def __init__(self):
        self.model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")
        self.num_faces = 0

    def detect_face_dnn(self, frame):
        face_bounding_box = []
        faces_bounding_box = []
        self.num_faces = 0

        # Get height and width of frame.
        (h, w) = frame.shape[:2]

        # Preprocess frame by resizing and converting to a blob.
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Feed the blob as input to the DNN face detection model.
        self.model.setInput(blob)
        detections = self.model.forward()

        # Iterate through each detection, get confidence score, draw a rectangle around each face.
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Use only the detections above the treshold.
            if confidence > 0.5:
                self.num_faces += 1
                # Get the bounding box for the face.
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                face_bounding_box = [x1, y1, x2, y2]

                # Draw a rectangle around the face.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'[{confidence:.2f}]', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                faces_bounding_box.append(face_bounding_box)

        return frame, faces_bounding_box

    def print_num_faces(self, frame):
        # Print number of detected faces.
        cv2.putText(frame, f"NUMBER OF DETECTED FACES: {self.num_faces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame