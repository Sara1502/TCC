import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt
import kagglehub



model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1")
movenet = model.signatures['serving_default']

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow("Movenet Multipose", frame)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print(frame)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BAYER_BGGR2RGB))

