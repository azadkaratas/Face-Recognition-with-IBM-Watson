import numpy as np
import cv2
import os
import zipfile

# In this code, you detects and crops the face images using OpenCV Haar Cascade.
# Then zip your images for suitable use of IBM Watson.

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
zf = zipfile.ZipFile("person1.zip", "w") #change zipFileName for each person you collect 

data_number=0

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #If face detected:
    if len(faces)>0 :
        #Crop and resize that face
        cv2.rectangle(img,(faces[0,0],faces[0,1]),(faces[0,0]+faces[0,2],faces[0,1]+faces[0,3]),(255,0,0),2) #Rectangle that face
        roi_color = img[faces[0,1]:faces[0,1]+faces[0,3], faces[0,0]:faces[0,0]+faces[0,2]]
        roi_color = cv2.resize(roi_color,(225,225))

        #Create jpg file and zip the files
        data_number+=1
        name='/photo'+str(data_number)+'.jpg'
        cv2.imwrite(name, roi_color, [cv2.IMWRITE_JPEG_QUALITY, 90])
        zf.write(name)
        if data_number==50:
            break #breaks when collects sufficient number of images
      
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

zf.close()
cap.release()
cv2.destroyAllWindows()
