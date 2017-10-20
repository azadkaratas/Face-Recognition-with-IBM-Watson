from watson_developer_cloud import VisualRecognitionV3
import json
from os.path import join, dirname
from os import environ

visual_recognition = VisualRecognitionV3('2016-05-20', api_key='YOUR_API_KEY_FROM_IBM_WATSON')#2016-05-20 is the version.

# This example is written for 4 people. You can add or remove as you wish.
with open(join(dirname('_file_'), 'person1.zip'), 'rb') as person1,\
    open(join(dirname('_file_'), 'person2.zip'), 'rb') as person2,\
    open(join(dirname('_file_'), 'person3.zip'), 'rb') as person3,\
    open(join(dirname('_file_'), 'person4.zip'), 'rb') as person4: 
    print(json.dumps(visual_recognition.create_classifier('faces_classifier_name', person1_positive_examples=person1, person2_positive_examples=person2, person3_positive_examples=person3, person4_positive_examples=person4), indent=2))

#Keep the returned face classifier ID to use it during test. Or you can find in the http://visual-recognition-tooling.mybluemix.net/.
