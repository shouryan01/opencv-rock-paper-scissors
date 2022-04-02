# Required Library Imports
from cv2 import cv2
import pickle
import warnings
warnings.filterwarnings("ignore")

# Importing Helper Classes
import detect_hands
from generate_data import num_hands
from make_calculations import calculate
from helper import classify_class

# Settings For The Text
position = (1000, 50)
font = cv2.FONT_HERSHEY_TRIPLEX
fontScale = 2
color = (0, 255, 0) # blue-green-red
thickness = 2
type = cv2.LINE_AA

hands = detect_hands.hand_detector(max_hands = num_hands)
model = pickle.load(open('model.sav','rb'))
cap = cv2.VideoCapture(0)

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
    prediction = model.predict([all_distance])[0]
    print(prediction)
    prediction_text = classify_class(prediction)
    image = cv2.putText(
      image, 
      prediction_text, 
      position, 
      font, 
      fontScale, 
      color, 
      thickness,
      type
    )
  cv2.imshow('Hands', image)
  cv2.waitKey(1)