import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
from datetime import datetime
from collections import defaultdict
from threading import Thread
import time
import pyttsx3  # <--- NUEVA LIBRERÍA


# --- CLASE PARA ELIMINAR EL LAG (THREADING) ---
class CameraStream:
    def __init__(self, rtsp_url):
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
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# --- FUNCIÓN DE AUDIO EN HILO SEPARADO ---
# Variable para evitar que el audio se solape
audio_playing = False


def play_audio_message():
    global audio_playing
    if audio_playing:
        return  # Si ya está hablando, no interrumpir ni acumular

    audio_playing = True
    try:
        # Inicializar el motor de voz dentro del hilo para evitar conflictos COM en Windows
        engine = pyttsx3.init()

        # Configuración opcional: intentar poner voz en español
        # voices = engine.getProperty('voices')
        # for voice in voices:
        #     if "spanish" in voice.name.lower():
        #         engine.setProperty('voice', voice.id)
        #         break

        engine.setProperty('rate', 150)  # Velocidad un poco más rápida
        engine.say("Intruso detectado, manos arriba y pantalones abajo")
        engine.runAndWait()
    except Exception as e:
        print(f"Error de audio: {e}")
    finally:
        audio_playing = False


def trigger_alert():
    # Lanzamos el audio en un hilo paralelo para no congelar el video
    t = Thread(target=play_audio_message)
    t.daemon = True  # El hilo muere si el programa principal muere
    t.start()


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


# --- CONFIGURACIÓN ---
RTSP_URL = 'rtsp://admin:admin@192.168.1.60/v_enc_000'  # Ajusta esto si es necesario
MODEL_PATH = 'yolov8n.pt'

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "TapoControl")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
CSV_SUMMARY = os.path.join(OUTPUT_DIR, "resumen_accesos.csv")

LINE_POSITION = 0.5  # Cambiado a 0.5 (CENTRO) según requerimiento
OFFSET = 0.05

# --- INICIALIZACIÓN ---
print("Cargando modelo...")
model = YOLO(MODEL_PATH)
track_history = defaultdict(lambda: [])
people_records = {}
crossed_ids = set()

# Control de Cooldown para alertas (para no spamear el audio)
last_alert_time = 0
ALERT_COOLDOWN = 5  # Segundos de espera entre alertas

if not os.path.exists(CSV_SUMMARY):
    with open(CSV_SUMMARY, 'w', newline='') as f:
        csv.writer(f).writerow(["ID", "Fecha", "Hora", "Evento", "Estancia_Seg"])

print("Conectando cámara...")
cam_stream = CameraStream(RTSP_URL).start()
time.sleep(1.0)

print("Sistema Iniciado. Audio activo para intrusos.")

while True:
    frame = cam_stream.read()
    if frame is None: continue

    h, w, _ = frame.shape
    line_x = int(w * LINE_POSITION)

    # Dibujar línea de cruce (Roja para indicar peligro/alerta)
    cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)

    results = model.track(frame, persist=True, verbose=False, classes=0, tracker="bytetrack.yaml", imgsz=640)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w_box, h_box = box
            tl = (int(x - w_box / 2), int(y - h_box / 2))
            br = (int(x + w_box / 2), int(y + h_box / 2))

            draw_rounded_rect(frame, tl, br, (255, 200, 0), 2, 20, 10)
            center_x = float(x)

            track = track_history[track_id]
            track.append((float(x), float(y + h_box / 2)))
            if len(track) > 20: track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(255, 200, 0), thickness=1)

            if track_id not in people_records:
                people_records[track_id] = {'start_time': datetime.now()}
            duration = (datetime.now() - people_records[track_id]['start_time']).total_seconds()

            # --- LÓGICA DE CRUCE Y AUDIO ---
            if len(track) > 2:
                start_x = track[0][0]
                event = None

                # Detectar cruce (Entrada o Salida)
                # Nota: Verificamos si NO ha cruzado antes (crossed_ids) para evitar doble registro
                # Pero para el audio, tal vez quieras que suene siempre que alguien cruce,
                # incluso si ya cruzó antes en otra dirección?
                # Aquí asumo que suena al cruzar la línea por primera vez en esa dirección.

                just_crossed = False

                # IZQUIERDA A DERECHA
                if start_x < line_x and center_x > line_x + (w * OFFSET):
                    if track_id not in crossed_ids:
                        event = "CRUCE_DERECHA"
                        just_crossed = True

                # DERECHA A IZQUIERDA
                elif start_x > line_x and center_x < line_x - (w * OFFSET):
                    if track_id not in crossed_ids:
                        event = "CRUCE_IZQUIERDA"
                        just_crossed = True

                if just_crossed:
                    crossed_ids.add(track_id)

                    # Registro CSV
                    with open(CSV_SUMMARY, 'a', newline='') as f:
                        csv.writer(f).writerow(
                            [track_id, datetime.now().strftime("%Y-%m-%d"),
                             datetime.now().strftime("%H:%M:%S"), event, round(duration, 2)])

                    # --- ACTIVAR AUDIO ---
                    # Verificamos cooldown global para no saturar
                    current_time = time.time()
                    if (current_time - last_alert_time) > ALERT_COOLDOWN:
                        print(f"¡ALERTA AUDIO! - ID: {track_id}")
                        trigger_alert()
                        last_alert_time = current_time

            label = f"ID: {track_id}"
            cv2.putText(frame, label, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    cv2.imshow("Monitor Almacen - Alerta Voz", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_stream.stop()
cv2.destroyAllWindows()