# twitter cyberbullying detection using machine learning

import pandas as pd
import numpy as np
import streamlit as st
import hashlib
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import cv2
from PIL import Image, ImageOps
import sqlite3
from streamlit_lottie import st_lottie
import requests
from tensorflow import keras

classifier = keras.models.load_model('model.h5')

conn = sqlite3.connect('database.db')
c = conn.cursor()

emotion_dict = {0:'Angry', 1 :'Disgust', 2: 'Fear', 3:'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


def classify_image(face):
    resu = []
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(face, 1.3, 5)
    for (x, y, w, h) in faces:
        face = face[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([face]) != 0:
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0) #48,48,1
            pred = classifier.predict(face)[0]
            pred = int(np.argmax(pred)) 
            final_pred = emotion_dict[pred] 
            output = str(final_pred)
            resu.append(output)

    return resu

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0) #(48,48,1)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)

            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()


def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def animation(value):
    r = requests.get(value)
    if r.status_code != 200:
        return None
    return r.json()

def plot(val):
    st_lottie(
            val,
            speed=1,
            reverse=False,
            loop=True,
            quality="High",
            #renderer="svg",
            height=400,
            width=-900,
            key=None,
        )

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def main():
    st.title("Real Time Facial Emotion Detection")
    menu = ["AdminPage", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "AdminPage":
        st.subheader("Admin Page")
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            if username == 'admin' and password == 'admin':
                st.success("Logged In as {}".format(username))
                st.subheader("View Users")
                user_result = view_all_users()
                clean_db = pd.DataFrame(user_result, columns=["User", "Password"])
                st.dataframe(clean_db)
            
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "Login":
        st.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                st.success("Logged In as {}".format(username))
                choice = st.selectbox("Select Mode", ["Live", "Photo"])
                if choice == "Live":
                    st.write("Click on start to use webcam and detect your face emotion")
                    st.error("You don't get emoji mapping experience in this mode")
                    st.success("Use Photo mode for emoji mapping experience")
                    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_transformer_factory=Faceemotion)
                elif choice == "Photo":
                    img = st.camera_input("Webcam", key="webcam")
                    if img is not None:
                        image = Image.open(img)
                        image = np.array(image)
                        res = classify_image(image)
                        if len(res) != 0:
                            st.write("Emotion : ", res[0])
                            if res[0] == 'Angry':
                                st.error("You are angry")
                                # show angry lottie url
                                anim = animation('https://assets5.lottiefiles.com/packages/lf20_tpv4t3bu.json')
                                plot(anim)

                            elif res[0] == 'Happy':
                                st.success("You are happy")
                                anim = animation('https://assets5.lottiefiles.com/packages/lf20_xv1gn5by.json')
                                plot(anim)

                            elif res[0] == 'Neutral':
                                st.write("You are neutral")
                                anim = animation('https://assets8.lottiefiles.com/packages/lf20_odh15kwd.json')
                                plot(anim)

                            elif res[0] == 'Sad':
                                st.write("You are sad")
                                anim = animation('https://assets10.lottiefiles.com/packages/lf20_hfnjm1i3.json')
                                plot(anim)

                            elif res[0] == 'Surprise':
                                st.success("You are surprise")
                                anim = animation('https://assets7.lottiefiles.com/datafiles/IzfSYsijBrigbf1/data.json')
                                plot(anim)

                            elif res[0] == 'Fear':
                                st.error("You are fear")
                                anim = animation('https://assets4.lottiefiles.com/private_files/lf30_qolzpdwh.json')
                                plot(anim)

                            elif res[0] == 'Disgust':
                                st.error("You are disgust")
                                anim = animation('https://assets5.lottiefiles.com/packages/lf20_9j6n7z6o.json')
                                plot(anim)

            else:
                st.warning("Incorrect Username/Password")
            
    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')

        if st.button("Signup"):
            create_usertable()
            hashed_new_password = make_hashes(new_password)
            add_userdata(new_user,hashed_new_password)
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")

if __name__ == '__main__':
    main()