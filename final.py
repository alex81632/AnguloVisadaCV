import cv2
import dlib
import math
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open('model.pkl', 'rb'))
import time

from sheets import write_columns_names, export_pandas_df_to_sheets, write_columns
from dotenv import load_dotenv
import os

load_dotenv()
file_path = "data.csv"

# Inicializar o detector de faces
detector = dlib.get_frontal_face_detector()

# Carregar o detector de pontos faciais (shape predictor)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Inicializar a captura de vídeo na maior resolução possível
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# printar a definicao da camera
print("Definicao da camera: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Imagem de teste para calibrar a distância focal
img_teste = cap.read()[1]
distancia_focal = 0.5 * img_teste.shape[1] / math.tan(60 / 2 * math.pi / 180)
centro = (img_teste.shape[1] // 2, img_teste.shape[0] // 2)
matriz_camera = np.array([[distancia_focal, 0, centro[0]],
                            [0, distancia_focal, centro[1]],
                            [0, 0, 1]], dtype=np.float32)
print(matriz_camera)

def calculate_image_and_model_points(pontos, image_points, model_points):

    # 2D image points
    image_points[0] = (pontos.part(30).x, pontos.part(30).y)     # Nose tip
    image_points[1] = (pontos.part(8).x, pontos.part(8).y)     # Chin
    image_points[2] = (pontos.part(36).x, pontos.part(36).y)     # Left eye left corner
    image_points[3] = (pontos.part(45).x, pontos.part(45).y)     # Right eye right corne
    image_points[4] = (pontos.part(48).x, pontos.part(48).y)     # Left Mouth corner
    image_points[5] = (pontos.part(54).x, pontos.part(54).y)     # Right mouth corner

    # 3D model points.
    model_points[0] = (0.0, 0.0, 0.0)             # Nose tip
    model_points[1] = (0.0, -330.0, -65.0)        # Chin
    model_points[2] = (-225.0, 170.0, -135.0)     # Left eye left corner
    model_points[3] = (225.0, 170.0, -135.0)      # Right eye right corne
    model_points[4] = (-150.0, -150.0, -125.0)    # Left Mouth corner
    model_points[5] = (150.0, -150.0, -125.0)     # Right mouth corner

    return image_points, model_points

def calcular_angulo_visada(pontos):
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (0, 0),     # Nose tip
                                (0, 0),     # Chin
                                (0, 0),     # Left eye left corner
                                (0, 0),     # Right eye right corne
                                (0, 0),     # Left Mouth corner
                                (0, 0)      # Right mouth corner
                            ], dtype="double")
    
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, 0.0, 0.0),             # Chin
                                (0.0, 0.0, 0.0),             # Left eye left corner
                                (0.0, 0.0, 0.0),             # Right eye right corne
                                (0.0, 0.0, 0.0),             # Left Mouth corner
                                (0.0, 0.0, 0.0),             # Right mouth corner
                            ])
    
    image_points, model_points = calculate_image_and_model_points(pontos, image_points, model_points)

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, matriz_camera, dist_coeffs)
    
    # printar o vetor de rotacao sem quebra de linha
    # print("Vetor de Rotacao: {0}, {1}, {2}".format(rotation_vector[0], rotation_vector[1], rotation_vector[2]))

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, matriz_camera, dist_coeffs)

    return image_points, nose_end_point2D, rotation_vector, translation_vector

def enviar_dados():
    # exportar atencao por id como csv de uma linha
    media_atencao = 0
    data = [0]
    data.insert(0, media_atencao)

    columns = [f"id{i}" for i in range(len(data) - 1)]
    columns.insert(0, "media")
    pd.DataFrame([data], columns=columns).to_csv("data.csv", index=False)

    
    if os.getenv("ID") is None:
        spreadsheet_id = write_columns_names(file_path)
        with open(".env", "a") as file:
            file.write(f"ID={spreadsheet_id}")

        export_pandas_df_to_sheets(spreadsheet_id, file_path)
    else:
        spreadsheet_id = os.getenv("ID")
        write_columns(file_path, spreadsheet_id)
        export_pandas_df_to_sheets(spreadsheet_id, file_path) 

while True:
    # Ler o frame da câmera
    ret, frame = cap.read()

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar as faces no frame
    faces = detector(gray)

    atencao_por_id = [] 

    for face in faces:
        # Detectar os pontos faciais (landmarks) na face
        shape = predictor(gray, face)

        image_points, nose_end_point2D, rotation_vector, translation_vector = calcular_angulo_visada(shape)

        x_adjusted = rotation_vector[0]+math.pi

        if x_adjusted > math.pi:
            x_adjusted = x_adjusted - 2*math.pi

        # salvar o x_adjusted no rotatio_vector
        rotation_vector[0] = x_adjusted
        
        columns = ['x', 'y', 'z']
        x, y, z = rotation_vector[0][0], rotation_vector[1][0], rotation_vector[2][0]
        df = pd.DataFrame([[x, y, z]], columns=columns)
        atencao = model.predict_proba(df)[0][1]

        atencao_por_id.append(atencao)
        
        # printar a porcentagem de atencao em cima da cabeca
        cv2.putText(frame, str(round(atencao*100, 2))+"%", ((face.left(), face.top())), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if(atencao>0.5):
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)


    media_atencao = np.mean(atencao_por_id)

    print("Media de atencao: ", media_atencao)

    # Verificar se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Mostrar o frame resultante invertida horizontalmente
    # cv2.imshow('Pontos Faciais', cv2.flip(frame, 1))
    cv2.imshow('Pontos Faciais', frame)


# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
