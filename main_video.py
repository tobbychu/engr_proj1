import cv2
from facenet_pytorch import MTCNN

from face_det_dnn import *
from facial_rec import *

def main():
    # Input video path
    input_video = "videos/bill_gates_3.mp4"

    # Set video_input=True for local video input, False for webcam input
    video_input = True
    # Set save_video=True to save result
    save_video = False

    # Load in video capture source
    if video_input:
        cap = cv2.VideoCapture(input_video)
    else:
        cap = cv2.VideoCapture(0)

    # Check video loading validity
    if not cap.isOpened():
        print("Error with camera or input video.")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create video writer if choose to save as video
    if save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    # Setup face detection models
    fddnn = FaceDetectionDNN()

    ### UNCOMMENT WHEN READY TO RUN FACIAL RECOGNITION ###
    fr = FacialRecognition()
    model, transform = fr.setup_model()

    while cap.isOpened():
        # Capture frame by frame
        ret, frame = cap.read()
        if not ret:
            print("Reach the end of the video. Completed.")
            break

        # Display FPS
        cv2.putText(frame, f"FPS: {round(fps, 1)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Face detection using DNN model
        frame, bounding_boxes = fddnn.detect_face_dnn(frame)
        frame = fddnn.print_num_faces(frame)

        print(f"bounding_box of face detection: {bounding_boxes}")
        print(f"----------")

        # Facial recognition
        ### UNCOMMENT WHEN READY TO RUN FACIAL RECOGNITION ###
        if bounding_boxes:
            frame, result = fr.recognize_face(model, transform, frame, bounding_boxes)
        frame = fr.print_num_faces_recognized(frame)
        

        # Display frame
        cv2.imshow('Frame', frame)

        # Save frame
        if save_video:
            out_writer.write(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release capture
    cap.release()
    cv2.destroyAllWindows()
    if save_video:
        out_writer.release()

if __name__ == "__main__":
    main()