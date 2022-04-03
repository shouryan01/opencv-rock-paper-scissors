# Required Library Imports
from cv2 import cv2
import pickle
import warnings
import random
import time
warnings.filterwarnings("ignore")

# Importing Helper Classes
import detect_hands
from generate_data import num_hands
from make_calculations import calculate
from helper import classify_class

# Settings For The Text
coordinates = (10,30)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.75
color = (255,0,0) #blue-green-red
thickness = 2
type = cv2.LINE_AA

hands = detect_hands.hand_detector(max_hands = num_hands)
model = pickle.load(open('../model.sav','rb'))
cap = cv2.VideoCapture(0)
timer = 0
prediction = None
computer_hand = None
prediction_text = None

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue
  image, list = hands.find_hand_landmarks(
    cv2.flip(frame, 1),
    draw_landmarks=False
  )
  if list:
    height, width, _ = image.shape
    all_distance = calculate(height,width, list)
    if timer % 90 == 0 and timer > 0:
      prediction = model.predict([all_distance])[0]
      computer_hand = random.randint(0,2)
      prediction_text = "The computer chose " + classify_class(computer_hand)
    if (prediction == 0 and computer_hand == 1) or (prediction == 1 and computer_hand == 2) or (
            prediction == 2 and computer_hand == 0):
      cv2.putText(
        image,
        "You Win!",
        (250, 250),
        font,
        fontScale,
        (0, 255, 0),
        thickness,
        type
      )
    elif (prediction == 1 and computer_hand == 0) or (prediction == 2 and computer_hand == 1) or (
            prediction == 0 and computer_hand == 2):
      cv2.putText(
        image,
        "You Lose!",
        (250, 250),
        font,
        fontScale,
        (0, 0, 255),
        thickness,
        type
      )
    elif(prediction == computer_hand and prediction != None and computer_hand != None):
      cv2.putText(
        image,
        "Draw!",
        (250, 250),
        font,
        fontScale,
        (255, 0, 0),
        thickness,
        type
      )
    image = cv2.putText(
      image,
      prediction_text,
      coordinates,
      font,
      fontScale,
      color,
      thickness,
      type
    )
  cv2.imshow('Hands', image)
  timer = timer + 1
  cv2.waitKey(1)