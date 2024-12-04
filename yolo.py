import cv2
from ultralytics import YOLO
import torch

# Verificar si hay GPU disponible y usarla si es posible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cargar el modelo YOLOv8 y moverlo al dispositivo adecuado
model = YOLO('yolov8n.pt')  # Modelo liviano (puedes usar otros como yolov8s.pt)
model.to(device)

# Inicializar la cámara (0 corresponde a la cámara por defecto del sistema)
cap = cv2.VideoCapture(0)

if not cap.isOpened(): 
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()  # Leer un frame de la cámara
    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        break

    # Realizar la predicción en el frame actual
    results = model(frame, conf=0.5)  # Establecer un umbral de confianza (confidence threshold)

    # Dibujar las cajas delimitadoras y etiquetas en el frame
    annotated_frame = results[0].plot()  # results[0] contiene las detecciones del primer frame

    # Mostrar el frame con las detecciones
    cv2.imshow("Detección en Tiempo Real - YOLOv8", annotated_frame)

    # Salir si el usuario presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()