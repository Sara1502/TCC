import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from datetime import datetime




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
    input_height, input_width =  192, 192 # Ajuste para o tamanho da entrada do modelo
    shaped = np.squeeze(np.multiply(keypoints, [input_height, input_width, 1]))

    # Ajuste os keypoints para o tamanho real do frame
    shaped[:, 0] *= y / input_height  # Ajusta a altura dos keypoints
    shaped[:, 1] *= x / input_width   # Ajusta a largura dos keypoints


    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    input_height, input_width = 192, 192  # Ajuste para o tamanho da entrada do modelo
    shaped = np.squeeze(np.multiply(keypoints, [input_height, input_width, 1]))

    # Ajuste os keypoints para o tamanho real do frame
    shaped[:, 0] *= y / input_height  # Ajusta a altura dos keypoints
    shaped[:, 1] *= x / input_width   # Ajusta a largura dos keypoints
        
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


def calcular_angulos(pose):
        """Calcula ângulos articulares para uma pose"""
        angulos = {}
        
        # Cotovelos
        if pose[5][2] > 0.3 and pose[7][2] > 0.3 and pose[9][2] > 0.3:
            angulos['cotovelo_esq'] = calcular_angulo_3pontos(pose[5], pose[7], pose[9])
        if pose[6][2] > 0.3 and pose[8][2] > 0.3 and pose[10][2] > 0.3:
            angulos['cotovelo_dir'] = calcular_angulo_3pontos(pose[6], pose[8], pose[10])
        
        # Joelhos
        if pose[11][2] > 0.3 and pose[13][2] > 0.3 and pose[15][2] > 0.3:
            angulos['joelho_esq'] = calcular_angulo_3pontos(pose[11], pose[13], pose[15])
        if pose[12][2] > 0.3 and pose[14][2] > 0.3 and pose[16][2] > 0.3:
            angulos['joelho_dir'] = calcular_angulo_3pontos(pose[12], pose[14], pose[16])
        
        # Adicione outras articulações conforme necessário
        return angulos


def calcular_angulo_3pontos(a, b, c):
    """Calcula ângulo em graus entre os pontos a-b-c"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def avaliarSincroniaGrupo(keypoints, limiar_percent=10):
    if keypoints is None or keypoints.shape[0] < 2:
        return 100  # Retorna 100% se não há pessoas suficientes para comparar

    membros_para_analise = [
        (10, 8),  # pé direito até joelho direito
        (9, 7),   # pé esquerdo até joelho esquerdo
        (8, 6),   # joelho direito até quadril direito
        (7, 5),   # joelho esquerdo até quadril esquerdo
        (12, 14), # ombro direito até cotovelo direito
        (11, 13), # ombro esquerdo até cotovelo esquerdo
        (14, 16), # cotovelo direito até punho direito
        (13, 15), # cotovelo esquerdo até punho esquerdo
    ]
    
    penalidade_total = 0
    num_comparacoes = 0
    
    for junta_a, junta_b in membros_para_analise:
        coeficientes = []
        for pessoa in keypoints:
            if (pessoa[junta_a][2] > 0.3 and pessoa[junta_b][2] > 0.3):
                dy = pessoa[junta_a][1] - pessoa[junta_b][1]
                dx = pessoa[junta_a][0] - pessoa[junta_b][0]
                if dx != 0:  # Evita divisão por zero
                    coeficiente = dy / dx
                    coeficientes.append(coeficiente)

        # Comparar todos os pares de coeficientes
        for i in range(len(coeficientes)):
            for j in range(i + 1, len(coeficientes)):
                avg = (abs(coeficientes[i]) + abs(coeficientes[j])) / 2
                diff_percent = abs(coeficientes[i] - coeficientes[j]) / (avg + 1e-6) * 100
                
                if diff_percent > limiar_percent:
                    penalidade = min(diff_percent - limiar_percent, 20)  # Penalidade máxima de 20% por comparação
                    penalidade_total += penalidade
                num_comparacoes += 1
    
    # Calcula a nota (100% - penalidade média)
    if num_comparacoes > 0:
        nota = max(0, 100 - (penalidade_total / num_comparacoes))
    else:
        nota = 100  # Se não houve comparações válidas, considera perfeito
    
    return nota




def calcular_confiabilidade_deteccao(keypoints_with_score):
    """Calcula a porcentagem de keypoints válidos detectados"""
    total_keypoints = 0
    keypoints_validos = 0
    
    for pessoa in keypoints_with_score:
        for kp in pessoa:
            total_keypoints += 1
            if kp[2] > 0.3:  # Confiança > 30%
                keypoints_validos += 1
                
    return (keypoints_validos / total_keypoints) * 100 if total_keypoints > 0 else 0

def verificar_erros_graves(keypoints_with_score):
    """Detecta erros grosseiros como fusão de pessoas"""
    if len(keypoints_with_score) == 0:
        return True
    
    # Verifica proporções corporais absurdas
    for pessoa in keypoints_with_score:
        ombro_esq = pessoa[5]
        ombro_dir = pessoa[6]
        quadril_esq = pessoa[11]
        
        if ombro_esq[2] > 0.3 and ombro_dir[2] > 0.3 and quadril_esq[2] > 0.3:
            largura_ombros = abs(ombro_esq[1] - ombro_dir[1])
            altura_tronco = abs(ombro_esq[0] - quadril_esq[0])
            
            # Razão inválida (indicando fusão de pessoas)
            if largura_ombros / (altura_tronco + 1e-6) > 2.5:
                return True
                
    return False






def detectar_acrobacia(keypoints_with_score, frame_shape, frame=None, debug=False, output_dir="debug_frames", frame_id=None): 
    altura_frame = frame_shape[0]
    largura_frame = frame_shape[1]
    caixa_altura = int(altura_frame * 0.50)
    topo_caixa = altura_frame - caixa_altura
    margem_lateral = int(largura_frame * 0.15)

    if debug and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    debug_frame = frame.copy() if (debug and frame is not None) else None
    if debug_frame is not None:
        # desenha a caixa
        cv2.rectangle(debug_frame, 
                     (margem_lateral, topo_caixa),
                     (largura_frame - margem_lateral, altura_frame),
                     (0, 255, 255), 2)

        # salva a imagem com a caixa desenhada
        nome_caixa = f"{output_dir}/caixa_{frame_id or datetime.now().strftime('%H%M%S%f')}.jpg"
        sucesso = cv2.imwrite(nome_caixa, debug_frame)
        print(f"[DEBUG] Tentativa de salvar imagem da caixa: {nome_caixa} — {'OK' if sucesso else 'FALHOU'}")

    for pessoa in keypoints_with_score:
        pe_esq = pessoa[15] if pessoa[15][2] > 0.6 else None
        pe_dir = pessoa[16] if pessoa[16][2] > 0.6 else None
        mao_esq = pessoa[9] if pessoa[9][2] > 0.5 else None
        mao_dir = pessoa[10] if pessoa[10][2] > 0.5 else None
        quadril_esq = pessoa[11] if pessoa[11][2] > 0.5 else None
        quadril_dir = pessoa[12] if pessoa[12][2] > 0.5 else None

        # Verifica PULO
        if pe_esq is not None and pe_dir is not None:
            pe_esq_no_ar = pe_esq[0] < topo_caixa
            pe_dir_no_ar = pe_dir[0] < topo_caixa
            altura_media_pes = (topo_caixa - pe_esq[0] + topo_caixa - pe_dir[0]) / 2
            quadril = quadril_esq if quadril_esq is not None else quadril_dir
            if pe_esq_no_ar and pe_dir_no_ar and altura_media_pes > 20 and quadril is not None and quadril[0] < topo_caixa:
                return True

        # Verifica NO CHÃO
        mao_esq_na_caixa = (
            mao_esq is not None and 
            mao_esq[0] >= topo_caixa and 
            margem_lateral <= mao_esq[1] <= (largura_frame - margem_lateral)
        )

        mao_dir_na_caixa = (
            mao_dir is not None and 
            mao_dir[0] >= topo_caixa and 
            margem_lateral <= mao_dir[1] <= (largura_frame - margem_lateral)
        )

        joelho_esq = pessoa[13] if pessoa[13][2] > 0.5 else None
        joelho_dir = pessoa[14] if pessoa[14][2] > 0.5 else None

        joelho_baixo = (
            (joelho_esq is not None and joelho_esq[0] >= topo_caixa) or
            (joelho_dir is not None and joelho_dir[0] >= topo_caixa)
        )

        if (mao_esq_na_caixa or mao_dir_na_caixa) and joelho_baixo:

            if debug_frame is not None:
                cv2.putText(debug_frame, "NO CHAO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                nome = f"{output_dir}/chao_{frame_id or datetime.now().strftime('%H%M%S%f')}.jpg"
                cv2.imwrite(nome, debug_frame)
            return True

    if debug_frame is not None:
        cv2.putText(debug_frame, "SEM DETECCAO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        nome = f"{output_dir}/sem_deteccao_{frame_id or datetime.now().strftime('%H%M%S%f')}.jpg"
        cv2.imwrite(nome, debug_frame)

    return False



def detectarTrocaPosicao(keypoints, frame_shape, threshold_overlap=0.3, threshold_speed=0.1):
    """
    Detecta trocas de posição incluindo sobreposição e movimento rápido.
    Corrige o erro de verificação de arrays NumPy.
    """
    if len(keypoints) < 2:
        return False

    # 1. Calcula bounding boxes e centros
    bboxes = []
    centros = []
    for pessoa in keypoints:
        # Verificação CORRETA de keypoints dos quadris
        quadril_esq = pessoa[11] if pessoa[11][2] > 0.3 else None
        quadril_dir = pessoa[12] if pessoa[12][2] > 0.3 else None
        
        # Modificação crucial: verifica se os arrays existem antes de usar
        if quadril_esq is not None and quadril_dir is not None:
            centro_x = (quadril_esq[1] + quadril_dir[1]) / 2
            centro_y = (quadril_esq[0] + quadril_dir[0]) / 2
            centros.append((centro_x, centro_y))
            
            # Bounding box baseada em todos os keypoints válidos
            pontos_validos = [kp[:2] for kp in pessoa if kp[2] > 0.3]
            if pontos_validos:
                x_coords = [p[1] for p in pontos_validos]
                y_coords = [p[0] for p in pontos_validos]
                bboxes.append((min(x_coords), min(y_coords), max(x_coords), max(y_coords)))

    # 2. Verifica sobreposição
    overlap_detectado = False
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            box1, box2 = bboxes[i], bboxes[j]
            # Cálculo da área de interseção
            x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
            y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
            area_overlap = x_overlap * y_overlap
            area_min = min((box1[2]-box1[0])*(box1[3]-box1[1]), 
                          (box2[2]-box2[0])*(box2[3]-box2[1]))
            if area_overlap/(area_min + 1e-6) > threshold_overlap:
                overlap_detectado = True
                break

    # 3. Verifica velocidade (com persistência entre frames)
    if not hasattr(detectarTrocaPosicao, 'centros_anteriores'):
        detectarTrocaPosicao.centros_anteriores = centros
    
    speed_detectado = False
    if centros and detectarTrocaPosicao.centros_anteriores:
        if len(centros) == len(detectarTrocaPosicao.centros_anteriores):
            speeds = []
            diagonal = np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
            for (x1, y1), (x2, y2) in zip(detectarTrocaPosicao.centros_anteriores, centros):
                speed = np.sqrt((x2-x1)**2 + (y2-y1)**2) / diagonal
                speeds.append(speed)
            if np.mean(speeds) > threshold_speed:
                speed_detectado = True
    
    detectarTrocaPosicao.centros_anteriores = centros  # Atualiza para o próximo frame
    return overlap_detectado or speed_detectado




framesComErro = []
totalFrames = 0
angulosPorFrame = []  
framesAcrobacias = 0
framesAssincronia = 0
framesTrocaPosicao = 0

#LOOP PRICIPAL
cap = cv2.VideoCapture('Kpop-Dance-Practice\\7-pessoas\\O.O\\O.O.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Processamento do frame (detecção com MoveNet)
    img = cv2.resize(frame, (192, 192))
    img_input = tf.expand_dims(img, axis=0)
    img_input = tf.cast(img_input, dtype=tf.int32)
    
    # Detecção de poses
    result = movenet(img_input)
    keypoints = result['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # Avaliação de sincronia
    notaSincronia = avaliarSincroniaGrupo(keypoints)
    angulosPorFrame.append(notaSincronia)

    # Detecção de acrobacias (e salva frames se necessário)
    isAcrobacia = detectar_acrobacia(keypoints, frame.shape)
    isTrocaPosicao = detectarTrocaPosicao(keypoints, frame.shape, threshold_overlap=0.3, threshold_speed=0.1)

    # Só avalia sincronia se NÃO for acrobacia NEM troca de posição
    if not isAcrobacia and not isTrocaPosicao:
        notaSincronia = avaliarSincroniaGrupo(keypoints)
        angulosPorFrame.append(notaSincronia)
        
        if notaSincronia < 80:
            fileName = f"frames/framesAssincronia/assincronia_{totalFrames:04d}.jpg"
            cv2.imwrite(fileName, frame.copy())
            framesAssincronia += 1
    else:
        if isAcrobacia:
            framesAcrobacias += 1
        if isTrocaPosicao:
            framesTrocaPosicao += 1

    totalFrames += 1

    print(f"Processado: {totalFrames}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
    



confiabilidadeModelo = np.mean([calcular_confiabilidade_deteccao(result['output_0'].numpy()[:,:,:51].reshape((6,17,3))) 
                               for _ in range(10)])  # Teste com 10 amostras



# Cálculo da nota final (ignorando frames de acrobacia):
framesValidos = totalFrames - framesAcrobacias - framesTrocaPosicao
if framesValidos > 0:
    nota_final = np.mean(angulosPorFrame)
else:
    nota_final = 0

print("\n=== RELATÓRIO FINAL ===")
print(f"Nota de sincronia (excluindo acrobacias e trocas de posição): {nota_final:.2f}%")
print(f"Frames analisados: {framesValidos}/{totalFrames} ({(framesValidos/totalFrames)*100:.1f}% válidos)")
print(f"Frames de acrobacia: {framesAcrobacias} ({(framesAcrobacias/totalFrames)*100:.1f}%)")
print(f"Frames de troca de posição: {framesTrocaPosicao} ({(framesTrocaPosicao/totalFrames)*100:.1f}%)")