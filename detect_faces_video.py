# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

class capture_data():
    def load_model(self):
        print("[INFO] loading model...")
        self.graph_path = "model_paths/deploy.prototxt.txt"
        self.weights_path = "model_paths/res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.graph_path,self.weights_path)
        return self.net

    def capture_video(self):
        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=0).start()
        time.sleep(2.0)
        self.root_path = "C:/Users/ASUS/Desktop/Attendance_system/dataset/"
        self.name = input("Enter your name for registration")
        self.input_directory = os.path.join(self.root_path,self.name)
        if not os.path.exists(self.input_directory):
            os.makedirs(self.input_directory, exist_ok = 'True')
            self.no_of_frame = 1
            # loop over the frames from the video stream
            while self.no_of_frame <= 100:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=400)

                # grab the frame dimensions and convert it to a blob
                (self.h, self.w) = self.frame.shape[:2]
                self.blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

                # pass the blob through the network and obtain the detections and
                # predictions
                self.net = self.load_model()
                self.net.setInput(self.blob)
                self.detections = self.net.forward()

                # loop over the detections
                for i in range(0, self.detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with the
                    # prediction
                    self.confidence = self.detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if self.confidence < 0.95:
                        continue

                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    self.box = self.detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                    (self.startX, self.startY, self.endX, self.endY) = self.box.astype("int")

                    # draw the bounding box of the face along with the associated
                    # probability
                    self.text = "{:.2f}%".format(self.confidence * 100)
                    self.y = self.startY - 10 if self.startY - 10 > 10 else self.startY + 10
                    self.blurry = cv2.Laplacian(self.frame, cv2.CV_64F).var()
                    print(self.blurry)
                    if self.blurry >50 :
                        cv2.imwrite(os.path.join(self.input_directory,str(self.name) + str(self.no_of_frame) + '.jpg'),self.frame)
                        cv2.rectangle(self.frame, (self.startX, self.startY), (self.endX, self.endY),(0, 0, 255),2)
                        cv2.putText(self.frame, self.text, (self.startX, self.y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255),2)
                        self.no_of_frame+= 1
                # show the output frame
                cv2.imshow("Frame",self.frame)
                self.key = cv2.waitKey(1)
                if self.key == ord('q'):
                    return 0


            # do a bit of cleanup
            self.vs.stop()
            cv2.destroyAllWindows()
        



# def capture_video():
# 	# load our serialized model from disk
#     print("[INFO] loading model...")
#     net = cv2.dnn.readNetFromCaffe("model_paths/deploy.prototxt.txt","model_paths/res10_300x300_ssd_iter_140000.caffemodel")

#     # initialize the video stream and allow the cammera sensor to warmup
#     print("[INFO] starting video stream...")
#     vs = VideoStream(src=0).start()
#     time.sleep(2.0)

#     root_path = "C:/Users/ASUS/Desktop/Attendance_system/dataset/"
#     name = input("Enter your name for registration")
#     input_directory = os.path.join(root_path, name)
#     if not os.path.exists(input_directory):
#         os.makedirs(input_directory, exist_ok = 'True')
#     no_of_frame = 1
#     # loop over the frames from the video stream
#     while no_of_frame <= no_of_frame:
#         # grab the frame from the threaded video stream and resize it
#         # to have a maximum width of 400 pixels
#         frame = vs.read()
#         frame = imutils.resize(frame, width=400)

#         # grab the frame dimensions and convert it to a blob
#         (h, w) = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

#         # pass the blob through the network and obtain the detections and
#         # predictions
#         net.setInput(blob)
#         detections = net.forward()

#         # loop over the detections
#         for i in range(0, detections.shape[2]):
#             # extract the confidence (i.e., probability) associated with the
#             # prediction
#             confidence = detections[0, 0, i, 2]

#             # filter out weak detections by ensuring the `confidence` is
#             # greater than the minimum confidence
#             if confidence < confidence:
#                 continue

#             # compute the (x, y)-coordinates of the bounding box for the
#             # object
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # draw the bounding box of the face along with the associated
#             # probability
#             text = "{:.2f}%".format(confidence * 100)
#             y = startY - 10 if startY - 10 > 10 else startY + 10
#             blurry = cv2.Laplacian(frame, cv2.CV_64F).var()
#             print(blurry)
#             if blurry > blurry:
#                 cv2.imwrite(os.path.join(input_directory,str(name) + str(no_of_frame) + '.jpg'),frame)
#                 cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255),2)
#                 cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255),2)
#                 no_of_frame+= 1
#         # show the output frame
#         cv2.imshow("Frame",frame)
#         key = cv2.waitKey(2)
#         if key == ord('q'):
#             return 0

#     # do a bit of cleanup
#     cv2.destroyAllWindows()
#     vs.stop()




