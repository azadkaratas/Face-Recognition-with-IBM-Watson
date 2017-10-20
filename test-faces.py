from watson_developer_cloud import VisualRecognitionV3
import json
from os.path import join, dirname
from os import environ
import numpy as np
import cv2

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

visual_recognition = VisualRecognitionV3('2016-05-20', api_key='YOUR_API_KEY_FROM_IBM_WATSON') #2016-05-20 is the version.

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #If a face detected:
    if len(faces)>0 :
        roi_color = img[faces[0,1]:faces[0,1]+faces[0,3], faces[0,0]:faces[0,0]+faces[0,2]]
        roi_color = cv2.resize(roi_color,(225,225))

        #First write cropped face into a .jpeg file
        name='test.jpg'
        cv2.imwrite(name, roi_color, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
    with open(join(dirname('_file_'), 'test.jpg'), 'rb') as test:
        #Then classify the jpeg file
        jsonResponse = json.dumps(visual_recognition.classify(classifier_ids='returned_faces_classifier_ID',threshold=0.01,images_file=test), indent=2)

    #Json Parsing to get valuable infos
    jsonResponse=json.loads(jsonResponse)
    jsonData = jsonResponse["images"][0]
    jsonData = jsonData['classifiers'][0]
    jsonData = jsonData['classes']

    name=[]
    score=[]
    for item in jsonData:
        name.append(item.get("class"))
        score.append(item.get("score"))

    #Print the predicted person
    print (max(score),name[np.argmax(score)])

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
