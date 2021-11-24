import cv2
from random import randrange

#load pre-trained data on face frontals from opencv (haarcascade algorithm)
TrainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to read
#img = cv2.imread('1.jpg')

webcam = cv2.VideoCapture(0)

#iteration forever over frames
while True:
   #read current frame
   successfulFrameRead, frame = webcam.read()
   grayscaled_img= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   #Detect face
   face_coordinates = TrainedFaceData.detectMultiScale(grayscaled_img)
   #Draw rectangles around face
   for(x,y,w,h) in face_coordinates:
      cv2.rectangle(frame,(x,y), (w+x,h+y),((randrange(128),randrange(225),randrange(225))),7)
      
   cv2.imshow('Mafaz face detector',frame)
   key=cv2.waitKey(1)

   #if q press it will stop
   if key==81 or key==113:
      break

print("Code completed")