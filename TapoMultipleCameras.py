import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
from datetime import datetime
from collections import defaultdict
from threading import Thread
import time


# --- CLASE PARA LECTURA DE CÁMARA EN HILO (NO TOCAR) ---
class CameraStream:
    def __init__(self, rtsp_url, name):
        self.rtsp_url = rtsp_url
        self.name = name
        self.stream = cv2.VideoCapture(rtsp_url)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# --- FUNCIÓN GRÁFICA: RECTÁNGULO REDONDEADO ---
def draw_rounded_rect(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    w, h = x2 - x1, y2 - y1
    r = min(r, w // 2, h // 2)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# --- CONFIGURACIÓN GENERAL ---
# Definimos las cámaras con un nombre amigable
CAMERAS_CONFIG = [
    {"id": "Taller", "ip": "192.168.100.5"},
    {"id": "Almacen", "ip": "192.168.100.8"},
    {"id": "Acceso", "ip": "192.168.100.14"},
    {"id": "Aula", "ip": "192.168.100.10"}
]
BASE_RTSP = "rtsp://nixlab:Nix2022@{}/stream1"

# Carpetas y CSV
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "TapoControl_Multi")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
CSV_SUMMARY = os.path.join(OUTPUT_DIR, "registro_multicamara.csv")

if not os.path.exists(CSV_SUMMARY):
    with open(CSV_SUMMARY, 'w', newline='') as f:
        csv.writer(f).writerow(["Camara", "ID_Persona", "Fecha", "Hora", "Evento", "Estancia_Seg"])

# --- INICIALIZACIÓN DE MODELOS Y STREAMS ---
streams = []
models = {}  # Diccionario de modelos independientes
track_histories = {}  # Diccionario de historiales independientes
people_records = {}
crossed_ids = {}  # Diccionario de IDs cruzados por cámara

print("Inicializando sistema multicámara...")

for cam_conf in CAMERAS_CONFIG:
    url = BASE_RTSP.format(cam_conf["ip"])
    cam_name = cam_conf["id"]

    # 1. Iniciar Stream en hilo
    print(f"Conectando a {cam_name} ({cam_conf['ip']})...")
    s = CameraStream(url, cam_name).start()
    streams.append(s)

    # 2. Cargar modelo independiente para esta cámara (para no mezclar tracking)
    # YOLOv8 Nano es ligero, cargar 4 instancias es viable en RAM (aprox 200MB total)
    models[cam_name] = YOLO('yolov8n.pt')

    # 3. Inicializar estructuras de memoria para esta cámara
    track_histories[cam_name] = defaultdict(lambda: [])
    people_records[cam_name] = {}
    crossed_ids[cam_name] = set()

time.sleep(2.0)  # Esperar a que todos los streams estabilicen buffer
print("¡Sistema Operativo! Presiona 'q' para salir.")

# Dimensiones para el grid (reducimos un poco para que quepan 4 en pantalla)
FRAME_W, FRAME_H = 640, 360

while True:
    processed_frames = []

    for stream in streams:
        frame = stream.read()
        cam_name = stream.name

        # Si una cámara falla, mostramos cuadro negro con aviso
        if frame is None:
            blank = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
            cv2.putText(blank, f"SIN SENAL: {cam_name}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            processed_frames.append(blank)
            continue

        # Redimensionar para uniformidad en el Grid
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        # Línea de control (Vertical al 50%)
        line_x = int(FRAME_W * 0.5)
        cv2.line(frame, (line_x, 0), (line_x, FRAME_H), (100, 100, 100), 1)

        # --- INFERENCIA CON EL MODELO ESPECÍFICO DE ESTA CÁMARA ---
        # conf=0.5 filtra detecciones dudosas
        results = models[cam_name].track(frame, persist=True, verbose=False, classes=0, tracker="bytetrack.yaml",
                                         imgsz=320)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w_box, h_box = box

                # Dibujo Estético
                tl = (int(x - w_box / 2), int(y - h_box / 2))
                br = (int(x + w_box / 2), int(y + h_box / 2))
                draw_rounded_rect(frame, tl, br, (255, 200, 0), 2, 15, 10)

                # Tracking y Lógica
                track = track_histories[cam_name][track_id]
                track.append((float(x), float(y + h_box / 2)))
                if len(track) > 15: track.pop(0)

                # Calcular Tiempos
                if track_id not in people_records[cam_name]:
                    people_records[cam_name][track_id] = datetime.now()

                duration = (datetime.now() - people_records[cam_name][track_id]).total_seconds()

                # Lógica de Cruce (Simplificada para ejemplo)
                if track_id not in crossed_ids[cam_name] and len(track) > 2:
                    start_x = track[0][0]
                    center_x = float(x)

                    event = None
                    if start_x < line_x and center_x > line_x + 20:
                        event = "ENTRADA"
                    elif start_x > line_x and center_x < line_x - 20:
                        event = "SALIDA"

                    if event:
                        crossed_ids[cam_name].add(track_id)
                        # REGISTRO CSV INCLUYENDO NOMBRE DE CÁMARA
                        with open(CSV_SUMMARY, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([cam_name, track_id, datetime.now().strftime("%Y-%m-%d"),
                                             datetime.now().strftime("%H:%M:%S"), event, round(duration, 2)])

                        # Flash visual en pantalla
                        cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 0) if event == "ENTRADA" else (0, 0, 255), -1)

                # Etiqueta
                cv2.putText(frame, f"ID:{track_id}", (tl[0], tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0),
                            1)

        # Nombre de la cámara en pantalla
        cv2.putText(frame, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        processed_frames.append(frame)

    # --- MONTAJE DEL GRID 2x2 ---
    # Aseguramos que tengamos 4 frames (rellenar si falta alguno por error)
    while len(processed_frames) < 4:
        processed_frames.append(np.zeros((FRAME_H, FRAME_W, 3), np.uint8))

    # Concatenamos: Arriba (0 y 1), Abajo (2 y 3)
    top_row = np.hstack((processed_frames[0], processed_frames[1]))
    bot_row = np.hstack((processed_frames[2], processed_frames[3]))

    # Grid Final
    combined_grid = np.vstack((top_row, bot_row))

    cv2.imshow("NIXLAB - CENTRAL DE MONITOREO 4 CAMARAS", combined_grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
for s in streams: s.stop()
cv2.destroyAllWindows()