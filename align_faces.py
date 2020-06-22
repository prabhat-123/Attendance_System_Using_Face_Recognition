# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=160)

training_dir = "C:/Users/ASUS/Desktop/Attendance_system/training_data/"
root_dir = "C:/Users/ASUS/Desktop/Attendance_system/dataset/"

input_name = input("Enter the name")
source_dir = os.path.join(root_dir,input_name)
i = 0
for imgs in os.listdir(source_dir):
    i+= 1
    img_array = cv2.imread(os.path.join(source_dir,imgs))
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    rects = detector(img_array,2)
    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, and then align the faces using facial landmarks
        (x,y,w,h) = rect_to_bb(rect)
        faceOrig = imutils.resize(img_array[y:y+h,x:x+w],width=256)
        faceAligned = fa.align(img_array,gray,rect)
        destination_path = os.path.join(training_dir,input_name)
        print(destination_path)
        print(os.path.join(destination_path + str(input_name)+str(i)+'.jpg'))
        cv2.imwrite(destination_path + str(input_name+ str(i) + '.jpg'),faceAligned)

        cv2.waitKey(0)


# # load the input image, resize it, and convert it to grayscale
# image = cv2.imread(args["image"])
# image = imutils.resize(image, width=800)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # show the original input image and detect faces in the grayscale
# # image
# cv2.imshow("Input", image)
# rects = detector(gray, 2)

# # loop over the face detections
# for rect in rects:
# 	# extract the ROI of the *original* face, then align the face
# 	# using facial landmarks
# 	(x, y, w, h) = rect_to_bb(rect)
# 	faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
# 	faceAligned = fa.align(image, gray, rect)

# 	import uuid
# 	f = str(uuid.uuid4())
# 	cv2.imwrite("foo/" + f + ".png", faceAligned)

# 	# display the output images
# 	cv2.imshow("Original", faceOrig)
# 	cv2.imshow("Aligned", faceAligned)
# 	cv2.waitKey(0)
