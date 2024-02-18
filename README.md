# engr_proj1
Authors: Tobby Zhu, Javier Farah

In this project we implemented a pipeline for real time facial recognition in videos. We performed transfer learning using Inception ResNet v1, which has been pre-trained on the VGGFace 2 dataset. We train the model on a dataset collected from two separate public datasets, CelebA and the Flickr-Faces-HQ Dataset, and manually labeled with two classes "bill_gates" and "not_bill_gates". We use stochastic gradient descent to train a logistical regression model with a sigmoid activation function.

The model achieves ~87% accuracy on validation set. We then use the model to perform facial recognition in videos. We use the DNN face detection model to output a bounding box around detected faces, and then feed the cropped face image to our facial recognition model to output either "bill_gates" or "not_bill_gates".

To run the program, install all dependencies in requirements.txt, then run main_video.py with Python 3.9 or higher. Change input_video variable in the main function in main_video.py to change the video used. Change video_input to switch between video file on disk and live webcam.