import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
from datetime import datetime
from collections import defaultdict

# --- CONFIGURACIÓN ---
RTSP_URL = 'rtsp://nixlab:Nix2022@192.168.100.8/stream1'
MODEL_PATH = 'yolov8n.pt'  # Se descargará solo la primera vez

# Carpetas
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "TapoControl")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CSV_SUMMARY = os.path.join(OUTPUT_DIR, "resumen_accesos.csv")
CSV_HEATMAP = os.path.join(OUTPUT_DIR, "datos_heatmap.csv")

# Configuración de la Línea de Entrada/Salida (Ajustar según tu cámara)
# Imaginemos una línea vertical en medio: Si cruza de Izq a Der es ENTRADA, viceversa SALIDA
# Valores entre 0 y 1 (proporción de la pantalla)
LINE_POSITION = 0.35
OFFSET = 0.05  # Zona muerta para evitar falsos positivos de gente parada en la línea

# --- INICIALIZACIÓN ---
print("Cargando modelo YOLOv8 Nano (Ligero)...")
model = YOLO(MODEL_PATH)

# Estructuras de datos para memoria
track_history = defaultdict(lambda: [])
people_records = {}  # Guardará {id: {'entry_time': timestamp, 'start_tick': time}}
crossed_ids = set()  # IDs que ya cruzaron la línea para no contarlos doble

# Crear CSVs
if not os.path.exists(CSV_SUMMARY):
    with open(CSV_SUMMARY, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Fecha", "Hora_Evento", "Tipo_Evento", "Duracion_Estancia_Seg"])

if not os.path.exists(CSV_HEATMAP):
    with open(CSV_HEATMAP, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "ID", "X", "Y"])  # X, Y normalizados para el mapa de calor

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Buffer bajo para tiempo real

print("Iniciando vigilancia...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtenemos dimensiones para dibujar la línea
    h, w, _ = frame.shape
    line_x = int(w * LINE_POSITION)

    # Dibujar línea de referencia (Amarilla)
    cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 2)
    cv2.putText(frame, "Linea de Control", (line_x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # --- DETECCIÓN Y TRACKING ---
    # persist=True es CLAVE para que recuerde los IDs entre frames
    # classes=0 fuerza a detectar SOLO personas
    results = model.track(frame, persist=True, verbose=False, classes=0, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Abrir el CSV de heatmap en modo append para escribir puntos rápidos
        with open(CSV_HEATMAP, 'a', newline='') as f_heat:
            writer_heat = csv.writer(f_heat)

            for box, track_id in zip(boxes, track_ids):
                x, y, w_box, h_box = box

                # Centroide de la persona (donde pisan, mejor para heatmaps)
                center_x = float(x)
                center_y = float(y + h_box / 2)

                # 1. Guardar Historial para visualización (Trayectoria)
                track = track_history[track_id]
                track.append((float(x), float(y)))  # Guardamos centro
                if len(track) > 30:  # Limitar largo de la cola visual
                    track.pop(0)

                # 2. Dibujar Trayectoria
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                # 3. Lógica de Entrada/Salida y Tiempos
                # Si es un ID nuevo, inicializamos su tiempo
                if track_id not in people_records:
                    people_records[track_id] = {'start_time': datetime.now()}

                current_time = datetime.now()
                duration = (current_time - people_records[track_id]['start_time']).total_seconds()

                # LOGICA DE CRUCE DE LÍNEA
                # Si cruza la línea y no ha sido registrado
                if track_id not in crossed_ids:
                    # Zona de Entrada (Izquierda a Derecha)
                    if center_x > line_x + (w * OFFSET) and track[0][0] < line_x:
                        event = "ENTRADA"
                        crossed_ids.add(track_id)
                        # Registrar en CSV
                        with open(CSV_SUMMARY, 'a', newline='') as f:
                            wr = csv.writer(f)
                            wr.writerow(
                                [track_id, current_time.strftime("%Y-%m-%d"), current_time.strftime("%H:%M:%S"), event,
                                 round(duration, 2)])
                        cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)  # Flash Verde

                    # Zona de Salida (Derecha a Izquierda)
                    elif center_x < line_x - (w * OFFSET) and track[0][0] > line_x:
                        event = "SALIDA"
                        crossed_ids.add(track_id)
                        # Registrar en CSV
                        with open(CSV_SUMMARY, 'a', newline='') as f:
                            wr = csv.writer(f)
                            wr.writerow(
                                [track_id, current_time.strftime("%Y-%m-%d"), current_time.strftime("%H:%M:%S"), event,
                                 round(duration, 2)])
                        cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)  # Flash Rojo

                # 4. Escribir datos para Heatmap (Normalizados 0-1 para facilitar gráficas luego)
                writer_heat.writerow(
                    [current_time.strftime("%H:%M:%S"), track_id, round(center_x / w, 4), round(center_y / h, 4)])

                # Visualización Info
                label = f"ID: {track_id} | {int(duration)}s"
                cv2.putText(frame, label, (int(x - w_box / 2), int(y - h_box / 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

    # Mostrar Frame
    cv2.imshow("Control de Almacen - YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()