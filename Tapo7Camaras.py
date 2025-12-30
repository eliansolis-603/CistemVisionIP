import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
from datetime import datetime
from collections import defaultdict
from threading import Thread
import time
import math


# --- CLASE PARA LECTURA DE CÁMARA EN HILO ---
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


# --- FUNCIÓN ESTÉTICA ---
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


# --- CONFIGURACIÓN DE LAS 7 CÁMARAS ---
# He renombrado los IDs duplicados para evitar errores en los diccionarios
CAMERAS_CONFIG = [
    {"id": "Taller", "ip": "192.168.100.5"},
    {"id": "Almacen", "ip": "192.168.100.8"},
    {"id": "Atras_1", "ip": "192.168.100.4"},
    {"id": "Atras_2", "ip": "192.168.100.45"},
    {"id": "Plasma", "ip": "192.168.100.6"},
    {"id": "Acceso", "ip": "192.168.100.14"},
    {"id": "Aula", "ip": "192.168.100.10"}
]

BASE_RTSP = "rtsp://nixlab:Nix2022@{}/stream1"

# Modelo (Usa .engine en Jetson para soporte de 7 cámaras)
# MODEL_FILE = 'yolov8n.engine'
MODEL_FILE = 'yolov8n.pt'  # Cámbialo a .engine en tu Jetson

# Carpetas
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "TapoControl_Multi")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
CSV_SUMMARY = os.path.join(OUTPUT_DIR, "registro_7camaras.csv")

if not os.path.exists(CSV_SUMMARY):
    with open(CSV_SUMMARY, 'w', newline='') as f:
        csv.writer(f).writerow(["Camara", "ID_Persona", "Fecha", "Hora", "Evento", "Estancia_Seg"])

# --- INICIALIZACIÓN ---
streams = []
models = {}
track_histories = {}
people_records = {}
crossed_ids = {}

print(f"Cargando {len(CAMERAS_CONFIG)} cámaras. Esto puede tardar unos segundos...")

for cam_conf in CAMERAS_CONFIG:
    url = BASE_RTSP.format(cam_conf["ip"])
    cam_name = cam_conf["id"]

    # Iniciar Stream
    print(f" -> Conectando {cam_name}...")
    s = CameraStream(url, cam_name).start()
    streams.append(s)

    # Cargar Modelo Independiente
    models[cam_name] = YOLO(MODEL_FILE)

    # Memoria
    track_histories[cam_name] = defaultdict(lambda: [])
    people_records[cam_name] = {}
    crossed_ids[cam_name] = set()

time.sleep(3.0)  # Esperar estabilización de buffers
print("¡SISTEMA ACTIVO!")

# Dimensiones de cada "miniatura" en el grid
# Reducimos tamaño para que quepan 7 en pantalla sin explotar la resolución
TILE_W, TILE_H = 480, 270

while True:
    processed_frames = []

    for stream in streams:
        frame = stream.read()
        cam_name = stream.name

        if frame is None:
            # Cuadro de error si falla la señal
            blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
            cv2.putText(blank, f"OFF: {cam_name}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            processed_frames.append(blank)
            continue

        # Resize temprano para mejorar rendimiento de dibujo y grid
        frame = cv2.resize(frame, (TILE_W, TILE_H))

        # Línea de control (Ajustada al nuevo tamaño)
        line_x = int(TILE_W * 0.5)
        cv2.line(frame, (line_x, 0), (line_x, TILE_H), (100, 100, 100), 1)

        # Inferencia
        # imgsz=320 es CRÍTICO para 7 cámaras. No lo subas a 640.
        results = models[cam_name].track(frame, persist=True, verbose=False, classes=0, tracker="bytetrack.yaml",
                                         imgsz=320)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w_box, h_box = box

                # Dibujo
                tl = (int(x - w_box / 2), int(y - h_box / 2))
                br = (int(x + w_box / 2), int(y + h_box / 2))
                draw_rounded_rect(frame, tl, br, (255, 200, 0), 2, 10, 5)

                # Tracking
                track = track_histories[cam_name][track_id]
                track.append((float(x), float(y + h_box / 2)))
                if len(track) > 15: track.pop(0)

                # Tiempos
                if track_id not in people_records[cam_name]:
                    people_records[cam_name][track_id] = datetime.now()
                duration = (datetime.now() - people_records[cam_name][track_id]).total_seconds()

                # Cruce de Línea
                if track_id not in crossed_ids[cam_name] and len(track) > 2:
                    start_x = track[0][0]
                    center_x = float(x)
                    event = None

                    # Umbral de cruce ajustado a resolución pequeña (10px)
                    if start_x < line_x and center_x > line_x + 10:
                        event = "ENTRADA"
                    elif start_x > line_x and center_x < line_x - 10:
                        event = "SALIDA"

                    if event:
                        crossed_ids[cam_name].add(track_id)
                        with open(CSV_SUMMARY, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([cam_name, track_id, datetime.now().strftime("%Y-%m-%d"),
                                             datetime.now().strftime("%H:%M:%S"), event, round(duration, 2)])

                        # Feedback visual
                        cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0) if event == "ENTRADA" else (0, 0, 255), -1)

                # Etiqueta
                cv2.putText(frame, f"ID:{track_id}", (tl[0], tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0),
                            1)

        # Etiqueta Nombre Cámara
        cv2.putText(frame, cam_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        processed_frames.append(frame)

    # --- LÓGICA DE GRID PARA 7 CÁMARAS ---
    # Necesitamos 8 espacios (4 columnas x 2 filas) para que quede parejo
    # Rellenamos con cuadros negros hasta tener múltiplo de 4
    while len(processed_frames) < 8:
        blank_slot = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        # Opcional: Poner logo de NixLab en los cuadros vacíos
        cv2.putText(blank_slot, "NIXLAB", (TILE_W // 2 - 50, TILE_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        processed_frames.append(blank_slot)

    # Armamos las filas (4 cámaras por fila)
    row1 = np.hstack(processed_frames[0:4])
    row2 = np.hstack(processed_frames[4:8])

    # Armamos el grid final
    grid = np.vstack((row1, row2))

    cv2.imshow("NIXLAB CCTV - 7 CAMARAS", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for s in streams: s.stop()
cv2.destroyAllWindows()