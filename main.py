import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from matplotlib import pyplot as plt
import kagglehub



model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1")
movenet = model.signatures['serving_default']



def loop_through_people(frame, keypoints_with_score, edges, confidence_threshold):
    for person in keypoints_with_score:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)



# Desenha keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)



# Desenha cantos
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


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(x2)), (0,0,255), 2)



cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Colocar a imagem no tamanho certo
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256,256)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detecção
    result = movenet(input_img)
    keypoints_with_score = result['output_0'].numpy()[:,:,:51].reshape((6, 17, 3))

    # Renderiza keypoints
    loop_through_people(frame, keypoints_with_score, EDGES, 0.3)

    cv2.imshow("Movenet Multipose", frame)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
