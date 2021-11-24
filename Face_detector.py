import cv2
from random import randrange

#load pre-trained data on face frontals from opencv (haarcascade algorithm)
TrainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to read
img = cv2.imread('1.jpg')

#rgb image ---->gray scale image
grayscaled_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detect face
face_coordinates = TrainedFaceData.detectMultiScale(grayscaled_img)




#Draw rectangles around face
for(x,y,w,h) in face_coordinates:
   cv2.rectangle(img,(x,y), (w+x,h+y),(randrange(128,256),randrange(128,255),randrange(128,153)),2)



#to show the image   
cv2.imshow('Mafaz face detector',img)

#wait for type any key to close the window
cv2.waitKey()

print("Code completed")