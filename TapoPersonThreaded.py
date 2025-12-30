import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
from datetime import datetime
from collections import defaultdict
from threading import Thread
import time


# --- CLASE PARA ELIMINAR EL LAG (THREADING) ---
class CameraStream:
    def __init__(self, rtsp_url):
        self.stream = cv2.VideoCapture(rtsp_url)
        # Bajar buffer interno de OpenCV al mínimo
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Inicia el hilo que lee frames constantemente
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            # Lee el frame, pero no lo devuelve todavía.
            # Al ser un bucle infinito rápido, siempre mantiene self.frame actualizado
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Devuelve el frame más reciente
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# --- FUNCIÓN ESTÉTICA: RECTÁNGULO REDONDEADO ---
def draw_rounded_rect(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Asegurar que el radio no sea mayor que la mitad del ancho/alto
    w = x2 - x1
    h = y2 - y1
    r = min(r, w // 2, h // 2)

    # Dibujar líneas rectas
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # Dibujar arcos (esquinas)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# --- CONFIGURACIÓN ---
RTSP_URL = 'rtsp://nixlab:Nix2022@192.168.100.5/stream1'
MODEL_PATH = 'yolov8n.pt'

# Carpetas
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "TapoControl")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
CSV_SUMMARY = os.path.join(OUTPUT_DIR, "resumen_accesos.csv")

# Línea de Control
LINE_POSITION = 0.3
OFFSET = 0.05

# --- INICIALIZACIÓN ---
print("Cargando modelo...")
model = YOLO(MODEL_PATH)
track_history = defaultdict(lambda: [])
people_records = {}
crossed_ids = set()

# Crear CSV si no existe
if not os.path.exists(CSV_SUMMARY):
    with open(CSV_SUMMARY, 'w', newline='') as f:
        csv.writer(f).writerow(["ID", "Fecha", "Hora", "Evento", "Estancia_Seg"])

# --- INICIO DEL STREAM SIN LAG ---
print("Conectando cámara en hilo paralelo...")
# Usamos nuestra clase personalizada en lugar de cv2.VideoCapture directo
cam_stream = CameraStream(RTSP_URL).start()
time.sleep(1.0)  # Dar tiempo a que el buffer inicial se llene

print("Sistema Iniciado. Presiona 'q' para salir.")

while True:
    # Obtenemos SIEMPRE el frame más nuevo
    frame = cam_stream.read()

    if frame is None:
        continue  # Si el hilo aún no lee nada, esperamos

    h, w, _ = frame.shape
    line_x = int(w * LINE_POSITION)

    # Dibujar línea (discreta)
    cv2.line(frame, (line_x, 0), (line_x, h), (100, 100, 100), 1)

    # --- TRACKING ---
    # imgsz=640 fuerza el tamaño de inferencia para velocidad
    results = model.track(frame, persist=True, verbose=False, classes=0, tracker="bytetrack.yaml", imgsz=640)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w_box, h_box = box

            # Coordenadas para dibujo (Top-Left y Bottom-Right)
            tl = (int(x - w_box / 2), int(y - h_box / 2))
            br = (int(x + w_box / 2), int(y + h_box / 2))

            # --- ESTÉTICA: RECTÁNGULO REDONDEADO ---
            # Color turquesa futurista, radio de 20px
            draw_rounded_rect(frame, tl, br, (255, 200, 0), 2, 20, 10)

            # Centroide
            center_x = float(x)

            # Tracking Visual (Cola)
            track = track_history[track_id]
            track.append((float(x), float(y + h_box / 2)))
            if len(track) > 20: track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(255, 200, 0), thickness=1)

            # Lógica de Tiempos
            if track_id not in people_records:
                people_records[track_id] = {'start_time': datetime.now()}

            duration = (datetime.now() - people_records[track_id]['start_time']).total_seconds()

            # Lógica de Cruce
            if track_id not in crossed_ids and len(track) > 2:
                # Comparamos posición actual vs hace unos frames (track[0]) para determinar dirección
                start_x = track[0][0]

                # ENTRADA
                if start_x < line_x and center_x > line_x + (w * OFFSET):
                    event = "ENTRADA"
                    crossed_ids.add(track_id)
                    with open(CSV_SUMMARY, 'a', newline='') as f:
                        csv.writer(f).writerow(
                            [track_id, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S"), event,
                             round(duration, 2)])

                # SALIDA
                elif start_x > line_x and center_x < line_x - (w * OFFSET):
                    event = "SALIDA"
                    crossed_ids.add(track_id)
                    with open(CSV_SUMMARY, 'a', newline='') as f:
                        csv.writer(f).writerow(
                            [track_id, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S"), event,
                             round(duration, 2)])

            # Etiqueta limpia arriba del cuadro
            label = f"Persona {track_id}"
            cv2.putText(frame, label, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    cv2.imshow("Monitor Almacen - Baja Latencia", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_stream.stop()
cv2.destroyAllWindows()