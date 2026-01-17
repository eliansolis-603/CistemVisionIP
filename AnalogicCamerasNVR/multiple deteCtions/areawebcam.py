import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ==========================================
# --- CONFIGURACIÓN PARA EXPOSICIÓN ---
# ==========================================
WEBCAM_INDEX = 0  # Cambia a 1 si no detecta la cámara correcta

# --- CALIDAD DE VIDEO ---
# Intentamos forzar Full HD para máxima calidad.
# Si la Jetson va lenta, baja a 1280 y 720.
WEBCAM_WIDTH = 1920
WEBCAM_HEIGHT = 1080

# --- ESTÉTICA VISUAL ---
# Qué tan transparentes son las zonas de color (0.0 invisible - 1.0 sólido)
# 0.15 es muy sutil, ideal para que no estorbe.
ALPHA_ZONAS = 0.15

# Modelos
print("Cargando modelo YOLOv8 Pose para demostración...")
# Usamos el modelo 'nano' para velocidad en tiempo real
model = YOLO('yolov8n-pose.pt')


# ==========================================
# CLASE DE VIDEO HILADO (Alta Calidad)
# ==========================================
class VideoCapturaHilada:
    def __init__(self, source, width, height):
        self.cap = cv2.VideoCapture(source)
        # Configurar la resolución deseada
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Imprimir resolución real lograda para verificar
        real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolución de cámara activa: {real_w}x{real_h}")

        self.q = queue.Queue(maxsize=1)
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
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.stop_thread = True
        self.thread.join()
        self.cap.release()


# ==========================================
# LÓGICA DE INTERFAZ GRÁFICA
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
            elif modo_dibujo in ['analisis', 'ergonomia'] and len(puntos_temp) == 4:
                zonas[modo_dibujo].append(np.array(puntos_temp, np.int32))
                puntos_temp = []
                modo_dibujo = None


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
print(f"Iniciando cámara en alta definición...")
stream = VideoCapturaHilada(WEBCAM_INDEX, WEBCAM_WIDTH, WEBCAM_HEIGHT)

window_name = "Demo Vision Artificial - Ergonomia y Analisis"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# Descomenta para pantalla completa en la exposición:
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, mouse_callback)

print("Sistema Listo. Controles: [1] Línea, [2] Zona Azul, [3] Zona Verde, [R] Reset, [Q] Salir")

while True:
    frame = stream.read()
    if frame is None:
        time.sleep(0.1)
        continue

    # 1. Inferencia IA
    results = model.track(frame, persist=True, verbose=False, classes=[0], tracker="bytetrack.yaml")

    # 2. Dibujar Zonas (Capa Transparente)
    overlay = frame.copy()

    # Líneas (Amarillo sólido)
    for l in zonas["lineas"]:
        cv2.line(frame, l[0], l[1], (0, 255, 255), 3)

    # Zonas rellenas en el overlay
    for poly in zonas["analisis"]:  # Azul
        cv2.fillPoly(overlay, [poly], (255, 100, 0))  # Azul BGR

    for poly in zonas["ergonomia"]:  # Verde
        cv2.fillPoly(overlay, [poly], (0, 200, 0))  # Verde BGR

    # Aplicar la transparencia sutil
    cv2.addWeighted(overlay, ALPHA_ZONAS, frame, 1 - ALPHA_ZONAS, 0, frame)

    # Dibujar contornos de zonas (sólidos para definición)
    for poly in zonas["analisis"]:
        cv2.polylines(frame, [poly], True, (255, 150, 0), 2)
    for poly in zonas["ergonomia"]:
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

    # 3. Procesamiento de Detecciones (SE DIBUJAN SOBRE LAS ZONAS)
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        keypoints = results[0].keypoints.xy.cpu().numpy()

        for box, track_id, kpts in zip(boxes, track_ids, keypoints):
            x1, y1, x2, y2 = map(int, box)
            centro_x, centro_y = int((x1 + x2) / 2), int(y2)

            en_zona_analisis = any(cv2.pointPolygonTest(p, (centro_x, centro_y), False) >= 0 for p in zonas["analisis"])
            en_zona_ergo = any(cv2.pointPolygonTest(p, (centro_x, centro_y), False) >= 0 for p in zonas["ergonomia"])

            # --- LÓGICA LEY SILLA (Zona Verde) ---
            if en_zona_ergo or (not zonas["ergonomia"] and not zonas["analisis"]):
                hip, knee, ankle = kpts[12], kpts[14], kpts[16]  # Pierna Derecha

                if np.sum(hip) > 0 and np.sum(knee) > 0:
                    angulo = calcular_angulo(hip, knee, ankle)
                    estado = "Sentado" if angulo < 140 else "De Pie"
                    # Colores más vivos para el texto de estado
                    color_status = (0, 50, 255) if estado == "Sentado" else (50, 255, 50)

                    if estado == "Sentado":
                        track_history[track_id]["sitting_time"] += 1 / 30
                    else:
                        track_history[track_id]["standing_time"] += 1 / 30

                    # Dibujar info con fondo negro para leerse mejor
                    label_estado = f"ID {track_id}: {estado}"
                    label_tiempos = f"Silla: {int(track_history[track_id]['sitting_time'])}s | Pie: {int(track_history[track_id]['standing_time'])}s"

                    # Etiquetas con fondo para mejor contraste
                    cv2.rectangle(frame, (x1, y1 - 40), (x1 + 200, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label_estado, (x1 + 5, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
                    cv2.putText(frame, label_tiempos, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1)

            # Dibujar Caja (Blanco brillante para resaltar)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            if en_zona_analisis:  # Indicador de zona azul
                cv2.circle(frame, (centro_x, centro_y), 8, (255, 100, 0), -1)
                cv2.circle(frame, (centro_x, centro_y), 10, (255, 255, 255), 2)

    # --- GUI INSTRUCCIONES (Esquina superior) ---
    modo_txt = f"MODO ACTIVO: {modo_dibujo.upper()}" if modo_dibujo else "NAVEGACION (Seleccione un modo con 1, 2 o 3)"
    # Fondo negro para el texto de instrucciones
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, modo_txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(puntos_temp) > 0:
        for pt in puntos_temp:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Punto {len(puntos_temp)}", (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)

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