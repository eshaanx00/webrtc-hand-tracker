import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp

mpHands = mp.solutions.hands
Hands = mpHands.Hands(min_detection_confidence=0.75,min_tracking_confidence=0.75,max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

def drawHands(img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hands.process(imgRGB)
        if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                        mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
        return img

def main():
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = drawHands(img=img)
            return img


    ctx = webrtc_streamer(key="Video", video_transformer_factory=VideoTransformer,media_stream_constraints={"video": True, "audio": False})

if __name__ == "__main__":
    main()
