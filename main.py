import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt
import kagglehub





# Substitua pelo caminho para a pasta que contém 'saved_model.pb' e 'variables'
caminho_para_o_modelo = "model"

# Carregar o modelo

model = tf.saved_model.load(caminho_para_o_modelo)
movenet = model.signatures['serving_default']

EDGES = {
    (0,1): 'm',
    (0,2): 'c',
    (1,3): 'm',
    (2,4): 'c',
    (0,5): 'm',
    (0,6): 'c',
    (5,7): 'm',
    (7,9): 'm',
    (6,8): 'c',
    (8,10): 'c',
    (5,6): 'y',
    (5,11): 'm',
    (6,12): 'c',
    (11,12): 'y',
    (11,13): 'm',
    (13,15): 'm',
    (12,14): 'c',
    (14,16):'c'
}

def loop_through_people(frame, keypoints_with_score, edges, confidence_threshold):
    for person in keypoints_with_score:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

cap = cv2.VideoCapture('Kpop-Dance-Practice\\8-pessoas\\Thanxx\\Thanxx.mp4')
while cap.isOpened():
    ret, frame = cap.read()

    img = frame.copy()
    input_height, input_width = 192, 192  # Ajuste de acordo com a resolução esperada
    img_resized = cv2.resize(img, (input_width, input_height))
    img_input = tf.expand_dims(img_resized, axis=0)  # Adiciona a dimensão do batch (1, height, width, 3)
    img_input = tf.cast(img_input, dtype=tf.int32)


    # Detection
    result = movenet(img_input)
    keypoints_with_score = result['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # Print keypoints for debugging (optional)
    print(keypoints_with_score)

    # Renderiza keypoints
    loop_through_people(frame, keypoints_with_score, EDGES, 0.3)

    cv2.imshow("Movenet Multipose", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
