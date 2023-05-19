import cv2
import dlib

# Inicializar o detector de faces
detector = dlib.get_frontal_face_detector()

# Carregar o detector de pontos faciais (shape predictor)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Ler o frame da câmera
    ret, frame = cap.read()

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar as faces no frame
    faces = detector(gray)

    for face in faces:
        # Detectar os pontos faciais (landmarks) na face
        shape = predictor(gray, face)

        # printar os 68 pontos faciais
        for i in range(0, 68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Mostrar o frame resultante
    cv2.imshow('Pontos Faciais', frame)

    # Verificar se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
