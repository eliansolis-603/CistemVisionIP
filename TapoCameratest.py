import cv2
import numpy as np
import os
import csv
from datetime import datetime
import time

# --- CONFIGURACIÓN ---
# Tu dirección RTSP
RTSP_URL = 'rtsp://nixlab:Nix2022@192.168.100.10/stream1'

# Configuración de carpetas y archivos
USER_PROFILE = os.path.expanduser("~")  # Detecta la carpeta de usuario en Windows/Linux/Mac
OUTPUT_DIR = os.path.join(USER_PROFILE, "Documents", "TapoTests")
CSV_FILE = os.path.join(OUTPUT_DIR, "registro_movimiento.csv")

# Umbral para considerar que hay movimiento (Sensibilidad)
SENSITIVITY_THRESHOLD = 25
# Cantidad mínima de píxeles cambiados para registrar evento (evita ruido)
MIN_AREA_PIXELS = 500
# Segundos de espera entre registros al CSV (para no spammear)
LOG_COOLDOWN = 1.0

# --- PREPARACIÓN DEL ENTORNO ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Carpeta creada: {OUTPUT_DIR}")

# Crear cabecera del CSV si no existe
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Fecha", "Hora", "Evento", "Intensidad (Px)"])

print(f"Conectando a: {RTSP_URL} ...")
cap = cv2.VideoCapture(RTSP_URL)

# Reducir el tamaño del buffer ayuda a reducir latencia en cámaras IP
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: No se pudo conectar a la cámara.")
    exit()

# Leemos el primer frame para iniciar la comparación
ret, frame_prev = cap.read()
if not ret:
    print("Error al leer el primer frame.")
    exit()

# Convertimos a gris y aplicamos un blur leve para eliminar ruido térmico de la cámara
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)

last_log_time = 0

print("Sistema iniciado. Presiona 'q' para salir.")

while True:
    ret, frame_curr = cap.read()
    if not ret:
        print("Error de conexión o stream finalizado.")
        break

    # 1. Preprocesamiento
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.GaussianBlur(gray_curr, (21, 21), 0)

    # 2. Calcular la diferencia absoluta (Diferencia de Frames)
    frame_diff = cv2.absdiff(gray_prev, gray_curr)

    # 3. Aplicar umbral (Threshold)
    # Si la diferencia es > 25, se vuelve blanco (255), si no, negro (0)
    _, mask = cv2.threshold(frame_diff, SENSITIVITY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Dilatar un poco la máscara para hacer los puntos más visibles/densos
    mask = cv2.dilate(mask, None, iterations=2)

    # 4. Visualización de "Densidad de Puntos"
    # En lugar de rectángulos, coloreamos los píxeles de movimiento en el frame original.
    # Donde la máscara es blanca, pintamos el pixel de Verde (0, 255, 0)
    # frame_curr[mask == 255] devuelve las coordenadas de movimiento
    frame_curr[mask == 255] = [0, 255, 0]

    # 5. Lógica de Registro (Logging)
    motion_pixels = np.count_nonzero(mask)

    current_time = time.time()

    if motion_pixels > MIN_AREA_PIXELS and (current_time - last_log_time) > LOG_COOLDOWN:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([date_str, time_str, "MOVIMIENTO DETECTADO", motion_pixels])

        # Indicador visual en consola
        print(f"[{time_str}] Movimiento registrado en CSV. Intensidad: {motion_pixels}")
        last_log_time = current_time

    # Actualizamos el frame anterior para la siguiente iteración
    gray_prev = gray_curr

    # Mostrar resultado
    # Redimensionamos un poco para que quepa bien en pantalla si es 1080p
    display_frame = cv2.resize(frame_curr, (960, 540))
    cv2.imshow('Deteccion de Movimiento - Puntos', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()