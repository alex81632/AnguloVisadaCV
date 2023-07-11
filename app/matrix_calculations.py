import cv2
import dlib
import math
import numpy as np
import pandas as pd
import pickle
import pygame as pg
model = pickle.load(open('model.pkl', 'rb'))
import time
import matplotlib.pyplot as plt

class AttentioRecognition:
    def __init__(self):

        # printar a media de atencao na tela
        pg.font.init()
        self.font = pg.font.SysFont('Comic Sans MS', 30)
        self.font_aviso = pg.font.SysFont('Comic Sans MS', 60)
        # Inicializar o detector de faces
        self.detector = dlib.get_frontal_face_detector()

        self.media_por_momento = []
        self.media_por_momento5 = []

        self.graph = pg.image.load("grafico.png")

        self.update_graf()

        self.max_alunos = 1

        self.shape = None
        self.atencao_por_id = [] 

        # Carregar o detector de pontos faciais (shape predictor)
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # Inicializar a captura de vídeo na maior resolução possível
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        img_teste = self.cap.read()[1]
        distancia_focal = 0.5 * img_teste.shape[1] / math.tan(60 / 2 * math.pi / 180)
        centro = (img_teste.shape[1] // 2, img_teste.shape[0] // 2)
        self.matriz_camera = np.array([[distancia_focal, 0, centro[0]],
                                    [0, distancia_focal, centro[1]],
                                    [0, 0, 1]], dtype=np.float32)
        
        self.media_atencao = 0
        self.media_5min = 0
        self.data_atencao = []
        
    def calculate_image_and_model_points(self, pontos, image_points, model_points):

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

    def calcular_angulo_visada(self):
        pontos = self.shape
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
        
        image_points, model_points = self.calculate_image_and_model_points(pontos, image_points, model_points)

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, self.matriz_camera, dist_coeffs)

        return rotation_vector
    
    def run(self):
        # Ler o frame da câmera
        ret, frame = self.cap.read()

        # Converter o frame para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar as faces no frame
        faces = self.detector(gray)

        self.atencao_por_id = []

        self.max_alunos = max(self.max_alunos, len(faces))

        for face in faces:
            # Detectar os pontos faciais (landmarks) na face
            self.shape = self.predictor(gray, face)

            rotation_vector = self.calcular_angulo_visada()

            x_adjusted = rotation_vector[0]+math.pi

            if x_adjusted > math.pi:
                x_adjusted = x_adjusted - 2*math.pi

            # salvar o x_adjusted no rotatio_vector
            rotation_vector[0] = x_adjusted
            
            columns = ['x', 'y', 'z']
            x, y, z = rotation_vector[0][0], rotation_vector[1][0], rotation_vector[2][0]
            df = pd.DataFrame([[x, y, z]], columns=columns)
            atencao = model.predict_proba(df)[0][1]

            self.atencao_por_id.append(atencao)
            
            # printar a porcentagem de atencao em cima da cabeca
            cv2.putText(frame, str(round(atencao*100, 2))+"%", ((face.left(), face.top())), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            if(atencao>0.5):
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)


        self.media_atencao = np.sum(self.atencao_por_id)/self.max_alunos
        # se for nan, colocar 0
        if np.isnan(self.media_atencao):
            self.media_atencao = 0
        
        self.media_por_momento.append((time.time(), self.media_atencao))
        
        self.data_atencao.append(self.media_atencao)

        if len(self.data_atencao) > 3000:
            self.data_atencao.pop(0)
        
        self.media_5min = np.mean(self.data_atencao)
        self.media_por_momento5.append((time.time(), self.media_5min))
        # se for nan, colocar 0
        if np.isnan(self.media_5min):
            self.media_5min = 0

        # fazer o grafico das duas atencoes em funcao do tempo
        figure = plt.figure(figsize=(16, 9))
        plt.plot([i[0] for i in self.media_por_momento], [i[1] for i in self.media_por_momento], label="Media de atencao")
        plt.plot([i[0] for i in self.media_por_momento5], [i[1] for i in self.media_por_momento5], label="Media de atencao nos ultimos 5 minutos")
        plt.legend()
        plt.xlabel("Tempo (s)")
        plt.ylabel("Media de atencao")
        plt.savefig("grafico.png")
        
        return frame
    
    def print_medias(self, screen):
        text = self.font.render(f"Atual: {round(self.media_atencao*100, 2)}%", False, (0, 0, 0))
        screen.blit(text, (20, 20))

        # printar a media de atencao na tela
        text = self.font.render(f"Media: {round(self.media_5min*100, 2)}%", False, (0, 0, 0))
        screen.blit(text, (20, 60))

    def retornar_media(self):
        return self.media_atencao, self.media_5min, self.atencao_por_id, self.max_alunos
    
    def aviso_baixa_atencao(self, screen):
        if self.media_5min < 0.3:
            text = self.font_aviso.render(f"Baixa atencao", False, (255, 0, 0))
            # aviso no centro da tela
            screen.blit(text, (screen.get_width()//2 - text.get_width()//2, screen.get_height()//2 - text.get_height()//2))

    def print_graf(self, screen, scale):
        if scale:
            self.graph = pg.transform.scale(self.graph, (screen.get_width(), screen.get_height()))
            screen.blit(self.graph, (0, 0))
        else:
            self.graph = pg.transform.scale(self.graph, (320, 180))
            # grafico no topo direito
            screen.blit(self.graph, (screen.get_width() - self.graph.get_width()-20, 20))

    def update_graf(self):
        self.graph = pg.image.load("grafico.png")