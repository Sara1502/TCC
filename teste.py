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

def loop_through_people(frame, keypoints_with_score, edges, confidence_threshold):
    for person in keypoints_with_score:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # Redimensiona as coordenadas dos keypoints

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)  # Desenha o ponto chave

EDGES = {
    (0, 1): (255, 0, 255),  # magenta
    (0, 2): (0, 255, 255),  # ciano
    (1, 3): (255, 0, 255),  # magenta
    (2, 4): (0, 255, 255),  # ciano
    (0, 5): (255, 255, 0),  # amarelo
    (0, 6): (0, 255, 0),    # verde
    (5, 7): (255, 0, 255),  # magenta
    (7, 9): (255, 0, 255),  # magenta
    (6, 8): (0, 255, 0),    # verde
    (8, 10): (0, 255, 0),   # verde
    (5, 6): (255, 255, 0),  # amarelo
    (5, 11): (255, 0, 255), # magenta
    (6, 12): (0, 255, 255), # ciano
    (11, 12): (255, 255, 0), # amarelo
    (11, 13): (255, 0, 255), # magenta
    (13, 15): (255, 0, 255), # magenta
    (12, 14): (0, 255, 255), # ciano
    (14, 16): (0, 255, 255)  # ciano
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # Redimensiona as coordenadas dos keypoints

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Desenha a linha de conexão

cap = cv2.VideoCapture('Kpop-Dance-Practice\\6-pessoas\\Crossroads\\Crossroads.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()
    input_height, input_width = 192, 192  # Ajuste de acordo com a resolução esperada
    img_resized = cv2.resize(img, (input_width, input_height))
    img_input = np.expand_dims(img_resized, axis=0)  # Adiciona a dimensão do batch (1, height, width, 3)
    img_input = img_input / 255.0  # Normaliza a imagem para o intervalo [0, 1]

    # Detecção
    result = movenet(img_input)
    keypoints_with_score = result['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))  # Formato dos keypoints

    # Renderiza keypoints e conexões
    loop_through_people(frame, keypoints_with_score, EDGES, 0.3)

    # Exibe o vídeo com as linhas de conexão
    cv2.imshow("Movenet Multipose", frame)

    # Encerra o loop ao pressionar a tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
