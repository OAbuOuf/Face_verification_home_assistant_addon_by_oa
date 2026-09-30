import os.path
from logging import raiseExceptions

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
import time
from collections import Counter

MATCH_THRESHOLD = 0.4
def getting_current_python_script_and_models_location():
    script_folder = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists('/data') == True:
        data_dir = '/data'
        with open("/data/options.json", "r") as file:
            mqtt_settings = json.load(file)
    else:

        data_dir = script_folder
        with open(os.path.join(script_folder, "mqtt-settings.json"), "r") as file:
            mqtt_settings = json.load(file)
    yunet_location = os.path.join(script_folder,'models','face_detection_yunet_2026may.onnx')
    sface_location = os.path.join(script_folder,'models','face_recognition_sface_2021dec.onnx')

    return mqtt_settings, data_dir, yunet_location,sface_location


#1
def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected with result:", reason_code)
    client.subscribe("face/enroll")
    client.subscribe("face/verify")

#2
def on_message(client, userdata, message):
    print(f'Topic receiveddd : {message.topic} , Payload receiveddd: {message.payload}')
    print(message.payload.decode("utf-8"))
    try:
        if message.topic == 'face/enroll':
            result = enroll_new_user(message.payload.decode("utf-8"))
        elif message.topic == 'face/verify':
            result = handle_verify()
        else:
            result = "TOPIC MISMATCH OR WRONG DATA"
    except Exception as e:
        result = f'ERROR FOUND: {e}'
    mqtt_client.publish('face/result', result)


#3
def capture_frames(number_of_captured_frames):
    captured_frames = []
    camera = mqtt_settings['camera_source']
    if camera.isdigit():
        camera = int(camera)
    open_the_camera = cv2.VideoCapture(camera)

    if open_the_camera.isOpened() == False:
        open_the_camera.release()
        raise Exception('CAMERA NOT REACHABLE...CHECK CAMERA POWER/CONNECTION')
#    time.sleep(0.2)
    for capture in range(number_of_captured_frames + 5):
        # Read the image
        # Capture one frame
        ret, frame = open_the_camera.read()
        if ret is False:
            print("ERROR!!!!!!!!!COULD NOT LOAD THE FRAME!!!!!!!!!")
            break
        else:
            captured_frames.append(frame)

    del captured_frames[0:4]
    # releasing the camera connection to not leave it 24/7 to allow other apps use the camera
    open_the_camera.release()

    return captured_frames
#4
def handle_verify():
    users_seen = []
    pictures = capture_frames(10)
    for every_picture in pictures:
        pictures_extracted = get_faces_and_face_features(every_picture)
        for features in pictures_extracted[1]:
            verified_face = verify_face(features)
            users_seen.append(verified_face[1])
    if len(users_seen) == 0:
        return 'NO FACES WERE FOUND....'
    else:
        counter = Counter(users_seen)
        user_found = counter.most_common(1)
        most_name, count_of_most_name = user_found[0]
        if most_name == 'unkown':
            return 'UNREGISTERED USER'
        elif count_of_most_name < 6:
            return 'UNKNOWN....CANT REASSURE ITS A REGISTERED USER'
        else:
            return f'REGISTERED USER FOUND: HELLO {most_name}'


#5
def enroll_new_user(name):
    name = name.strip()
    if name == '':
        return 'NO NAME ADDED PLEASE ADD NAME'
    else:
        pictures = capture_frames(10)
        for every_picture in pictures:
            picture_faces = get_faces_and_face_features(every_picture)
            if len(picture_faces[1]) == 1:
                    # TRYING MULTIPLE WAYS OF CALLING OUT FUNCTION
                    score_found, name_found = verify_face(extracted_features_face=picture_faces[1][0])
                    if name_found != 'unkown':
                        return f"YOU ARE ALREADY REGISTERED...WELCOME {name_found}"
                    else:
                        add_face(picture_faces[1][0],name)
                        return f"NEW USER ADDED...WELCOME {name}"

        return 'PLEASE TRY AGAIN...WITH ONE FACE PERSON ONLY IN VIEW TO REGISTER'







mqtt_settings, data_dir,yunet_location, sface_location = getting_current_python_script_and_models_location()


# load the yunet model for face recognition
yunet = cv2.FaceDetectorYN.create(yunet_location,"",(320,320))
# load the sface model for face verification
sface = cv2.FaceRecognizerSF.create(sface_location,"")




enrolled = []
enrolled_name = []
saved_database = os.path.join(data_dir, 'saved_file.npz')

#5
# loading database function
def load_saved_database(saved_database):
    if os.path.exists(saved_database):
        loaded_data = np.load(saved_database)
        enrolled.extend(list(loaded_data['features']))
        enrolled_name.extend(list(loaded_data['name']))
        loaded_data.close()
    else:
        pass
#6
def add_face(extracted_features, save_new_name):
        enrolled.append(extracted_features.copy())
        enrolled_name.append(save_new_name)
        save_database()
#7
#for later on
def delete_face():
    return None
#8
def rename_face():
    return None
#9
def save_database():
    saved_file = np.savez(saved_database, name=enrolled_name, features=enrolled)
    return saved_file

#10
def get_faces_and_face_features(frame):
    detected_faces = []
    detected_features = []
    frame_W = int(frame.shape[1])
    frame_H = int(frame.shape[0])
    yunet.setInputSize((frame_W, frame_H))
    _, faces = yunet.detect(frame)
    if faces is not None:
        for face in faces:
            aligned_face = sface.alignCrop(frame, face)
            extracted_features_face = sface.feature(aligned_face)
            detected_faces.append(face)
            detected_features.append(extracted_features_face)
    return detected_faces, detected_features
#11
def verify_face(extracted_features_face):
    best_score = 0
    best_name = ''
    for saved_fingerprint, saved_name in zip(enrolled, enrolled_name):
        score_difference = sface.match(saved_fingerprint, extracted_features_face, cv2.FaceRecognizerSF_FR_COSINE)
        if score_difference >= best_score:
            best_score = score_difference
            best_name = saved_name
    if best_score < MATCH_THRESHOLD:
        best_name = 'unkown'
    return best_score,best_name

load_saved_database(saved_database=saved_database)



# mqtt client
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

mqtt_client.username_pw_set(mqtt_settings['mqtt_username'],mqtt_settings['mqtt_password'])



mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(mqtt_settings['mqtt_host_ip'], mqtt_settings['mqtt_port'], 60)

mqtt_client.loop_forever()

