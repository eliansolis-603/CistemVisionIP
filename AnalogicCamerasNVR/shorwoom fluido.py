import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ==========================================
# --- CONFIGURACIÓN ---
# ==========================================
NVR_IP = "192.168.1.112"
USUARIO = "admin"
PASSWORD = "admin2025"
CANAL = "6"
SUBTYPE = "0"  # 0 = Main Stream (Alta Calidad)

# --- OPTIMIZACIÓN DE RENDIMIENTO ---
# Ejecutar IA cada X cuadros.
# 1 = Analiza odo (lento).
# 3 = Analiza 1, muestra 2 sin analizar (video fluido).
SKIP_FRAMES = 3

# --- ESTÉTICA ---
ALPHA_ZONAS = 0.15

print("Cargando modelo YOLOv8 Pose en GPU...")
# Aseguramos que corra en GPU si tienes CUDA, si no, usará CPU
model = YOLO('yolov8n-pose.pt')


# ==========================================
# CLASE DE VIDEO HILADO (Optimizado)
# ==========================================
class VideoCapturaHilada:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        # Reducir buffer interno de FFmpeg para minimizar lag de red
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.q = queue.Queue(maxsize=2)  # Aumentamos ligeramente buffer para suavizar
        self.stop_thread = False
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while not self.stop_thread:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Estrategia LIFO: Si la cola está llena, sacamos el viejo para meter el nuevo (video real-time)
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return self.q.get(timeout=1)  # Timeout para no congelar si se cae la red
        except queue.Empty:
            return None

    def release(self):
        self.stop_thread = True
        self.thread.join()
        self.cap.release()


# ==========================================
# LÓGICA DE DIBUJO E INTERFAZ
# ==========================================
puntos_temp = []
zonas = {"lineas": [], "analisis": [], "ergonomia": []}
modo_dibujo = None


def mouse_callback(event, x, y, flags, param):
    global puntos_temp, modo_dibujo
    if event == cv2.EVENT_LBUTTONDOWN:
        if modo_dibujo:
            puntos_temp.append((x, y))
            if modo_dibujo == 'linea' and len(puntos_temp) == 2:
                zonas["lineas"].append(puntos_temp.copy())
                puntos_temp = []
                modo_dibujo = None
                print("Línea creada.")
            elif modo_dibujo in ['analisis', 'ergonomia'] and len(puntos_temp) == 4:
                zonas[modo_dibujo].append(np.array(puntos_temp, np.int32))
                puntos_temp = []
                modo_dibujo = None
                print(f"Zona {modo_dibujo} guardada.")


def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle


track_history = defaultdict(lambda: {"sitting_time": 0, "standing_time": 0})

# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================
rtsp_url = f"rtsp://{USUARIO}:{PASSWORD}@{NVR_IP}:554/cam/realmonitor?channel={CANAL}&subtype={SUBTYPE}"
print(f"Conectando a: {rtsp_url} ...")

stream = VideoCapturaHilada(rtsp_url)
time.sleep(2.0)  # Buffer inicial

window_name = "Expo Vision Artificial"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, mouse_callback)

frame_count = 0
last_results = None  # Variable para guardar la inferencia anterior

while True:
    frame = stream.read()
    if frame is None:
        continue

    # --- OPTIMIZACIÓN: INFERENCIA INTERCALADA ---
    # Solo ejecutamos la IA cada 'SKIP_FRAMES' cuadros
    if frame_count % SKIP_FRAMES == 0:
        # Ejecutamos YOLO
        results = model.track(frame, persist=True, verbose=False, classes=[0], tracker="bytetrack.yaml")
        last_results = results  # Guardamos resultado

    frame_count += 1

    # --- VISUALIZACIÓN (Usamos 'last_results' siempre) ---
    overlay = frame.copy()

    # 1. Dibujar Zonas (Siempre visibles)
    for l in zonas["lineas"]: cv2.line(frame, l[0], l[1], (0, 255, 255), 3)
    for poly in zonas["analisis"]: cv2.fillPoly(overlay, [poly], (255, 100, 0))
    for poly in zonas["ergonomia"]: cv2.fillPoly(overlay, [poly], (0, 200, 0))
    cv2.addWeighted(overlay, ALPHA_ZONAS, frame, 1 - ALPHA_ZONAS, 0, frame)

    # Bordes zonas
    for poly in zonas["analisis"]: cv2.polylines(frame, [poly], True, (255, 150, 0), 2)
    for poly in zonas["ergonomia"]: cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

    # 2. Procesar Resultados de la IA (Si existen)
    if last_results and last_results[0].boxes.id is not None:
        boxes = last_results[0].boxes.xyxy.cpu().numpy()
        track_ids = last_results[0].boxes.id.int().cpu().numpy()
        keypoints = last_results[0].keypoints.xy.cpu().numpy()

        for box, track_id, kpts in zip(boxes, track_ids, keypoints):
            x1, y1, x2, y2 = map(int, box)
            centro_x, centro_y = int((x1 + x2) / 2), int(y2)

            en_zona_ergo = any(cv2.pointPolygonTest(p, (centro_x, centro_y), False) >= 0 for p in zonas["ergonomia"])

            # Logica Ley Silla
            if en_zona_ergo or (not zonas["ergonomia"] and not zonas["analisis"]):
                hip, knee, ankle = kpts[12], kpts[14], kpts[16]
                if np.sum(hip) > 0 and np.sum(knee) > 0:
                    angulo = calcular_angulo(hip, knee, ankle)
                    estado = "Sentado" if angulo < 140 else "De Pie"
                    color_status = (0, 50, 255) if estado == "Sentado" else (50, 255, 50)

                    # Solo actualizamos el tiempo REAL en los frames que hubo detección para no contar doble falso
                    if frame_count % SKIP_FRAMES == 0:
                        incremento = (1 / 30) * SKIP_FRAMES  # Compensamos el tiempo
                        if estado == "Sentado":
                            track_history[track_id]["sitting_time"] += incremento
                        else:
                            track_history[track_id]["standing_time"] += incremento

                    label_estado = f"ID {track_id}: {estado}"
                    label_tiempos = f"Silla: {int(track_history[track_id]['sitting_time'])}s | Pie: {int(track_history[track_id]['standing_time'])}s"

                    # Dibujo
                    cv2.rectangle(frame, (x1, y1 - 45), (x1 + 220, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label_estado, (x1 + 5, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
                    cv2.putText(frame, label_tiempos, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Indicador de detección (Visual feedback de que la IA funciona)
            if frame_count % SKIP_FRAMES == 0:
                cv2.circle(frame, (x2 - 10, y1 + 10), 5, (0, 255, 0), -1)  # Punto verde parpadeante en detección nueva

    # --- GUI ---
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    txt_info = f"MODO: {modo_dibujo if modo_dibujo else 'NAVEGACION'} | FPS Visual: Fluido | IA: 1/{SKIP_FRAMES}"
    cv2.putText(frame, txt_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if len(puntos_temp) > 0:
        for pt in puntos_temp:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        zonas = {"lineas": [], "analisis": [], "ergonomia": []}
    elif key == ord('1'):
        modo_dibujo = "linea"; puntos_temp = []
    elif key == ord('2'):
        modo_dibujo = "analisis"; puntos_temp = []
    elif key == ord('3'):
        modo_dibujo = "ergonomia"; puntos_temp = []

stream.release()
cv2.destroyAllWindows()