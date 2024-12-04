import cv2
from ultralytics import YOLO
import torch
import time

# Verificar si hay GPU disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')  # Cambiar a 'yolov8s.pt' si necesitas más precisión
model.to(device)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Variables para métricas
frame_count = 0
total_inference_time = 0
start_time = time.time()  # Tiempo de inicio para FPS

print("Presiona 'q' para salir.")

# Abrir archivo para guardar métricas
with open("metricasv2.txt", "w") as f:
    f.write("Metricas de Deteccion en Tiempo Real\n")
    f.write("====================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        frame_count += 1

        # Medir el tiempo de inferencia
        inference_start = time.time()
        results = model(frame, conf=0.5)  # Umbral de confianza del 50%
        inference_end = time.time()

        # Calcular el tiempo de inferencia del cuadro actual
        inference_time = inference_end - inference_start
        total_inference_time += inference_time

        # Dibujar las cajas delimitadoras y etiquetas
        annotated_frame = results[0].plot()

        # Mostrar el frame con las detecciones
        cv2.imshow("Detección en Tiempo Real - YOLOv8", annotated_frame)

        # Salir si el usuario presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calcular métricas
    total_time = time.time() - start_time
    fps = frame_count / total_time  # FPS promedio
    avg_inference_time = total_inference_time / frame_count  # Tiempo promedio de inferencia por cuadro

    # Escribir métricas en el archivo
    f.write(f"Total de cuadros procesados: {frame_count}\n")
    f.write(f"FPS promedio: {fps:.2f}\n")
    f.write(f"Tiempo promedio de inferencia por cuadro: {avg_inference_time:.4f} segundos\n")

    # Nota: La métrica mAP requiere evaluación en un conjunto de datos etiquetado
    f.write("\nNota: La métrica mAP requiere evaluación en un conjunto de datos etiquetado para ser calculada.")
    
print("Métricas guardadas en 'metricasv2.txt'.")

# Cerrar la cámara
cap.release()
cv2.destroyAllWindows()