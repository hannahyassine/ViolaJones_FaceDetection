import cv2 as cv
import os

# Input image path
image_path = 'D:\\DEV\\PYTHON\\ViolaJones_FaceDetection\\The Solvay Conference, 1927.jpg'

# Read image
input_image = cv.imread(image_path)

# Resize the input image to specific height and width
# to ensure any image that is put through this algorith is of the same size
width = 1600
height = 806

# stores the width and height in the tuple dim 
# a tuple is a collection that is ordered and unchangeable
dim = (width, height)

# resizes the image to the dimensions width and height.
resized_image = cv.resize(input_image, dim, interpolation = cv.INTER_AREA)

# prints the resized image dimensions to the console 
print('Resized Dimensions : ', resized_image.shape)

# outputs the resized image to the screen
cv.imshow("Resized image", resized_image)

# Convert RGB to Gray for Viola-Jones (algorithm requirment)
grayscale_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale Image', grayscale_image)

cascPathface = os.path.dirname(cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(cv.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv.CascadeClassifier(cascPathface)
eyeCascade = cv.CascadeClassifier(cascPatheyes)

while True:
    faces = faceCascade.detectMultiScale(grayscale_image,
                                         scaleFactor=1.1,
                                         minNeighbors=7,
                                         minSize=(50, 50),
                                         flags=cv.CASCADE_SCALE_IMAGE)
    
    for (x,y,w,h) in faces:
            cv.rectangle(grayscale_image , (x, y), (x + w, y + h),(255,255,255), 2)
            faceROI = grayscale_image [y:y+h,x:x+w]
            eyes = eyeCascade.detectMultiScale(faceROI)
            for (x2, y2, w2, h2) in eyes:
                eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
                radius = int(round((w2 + h2) * 0.10))
                grayscale_image  = cv.circle(grayscale_image , eye_center, radius, (255, 0, 0), 4)
    
    cv.imshow('Faces Detected', grayscale_image)
    
    #waits for user to press any key 
    cv.waitKey(0) 
    
    break

#closing all open windows 
cv.destroyAllWindows() 
