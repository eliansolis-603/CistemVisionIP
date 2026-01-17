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
# IP del NVR (accesible gracias a tu Mac-Puente)
NVR_IP = "192.168.1.108"
USUARIO = "admin"
PASSWORD = "admin2025"
CANAL = "2"
SUBTYPE = "0"

# Modelos
# Usamos el modelo POSE para detectar personas y sus articulaciones (Ley Silla)
print("Cargando modelo YOLOv8 Pose en GPU...")
model = YOLO('yolov8n-pose.pt')


# ==========================================
# CLASE DE VIDEO HILADO (Sin Lag)
# ==========================================
class VideoCapturaHilada:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
# LÓGICA DE GESTIÓN DE ZONAS E INTERFAZ
# ==========================================
puntos_temp = []
zonas = {"lineas": [], "analisis": [], "ergonomia": []}
modo_dibujo = None  # Puede ser 'linea', 'analisis', 'ergonomia'


def mouse_callback(event, x, y, flags, param):
    global puntos_temp, modo_dibujo

    if event == cv2.EVENT_LBUTTONDOWN:
        if modo_dibujo:
            puntos_temp.append((x, y))

            # Lógica para cerrar las figuras
            if modo_dibujo == 'linea' and len(puntos_temp) == 2:
                zonas["lineas"].append(puntos_temp.copy())
                puntos_temp = []
                modo_dibujo = None
                print("Línea creada.")

            elif modo_dibujo in ['analisis', 'ergonomia'] and len(puntos_temp) == 4:
                zonas[modo_dibujo].append(np.array(puntos_temp, np.int32))
                puntos_temp = []
                modo_dibujo = None
                print(f"Zona {modo_dibujo} creada.")


def calcular_angulo(a, b, c):
    """Calcula el ángulo entre tres puntos (Cadera, Rodilla, Tobillo)"""
    a = np.array(a)  # Cadera
    b = np.array(b)  # Rodilla
    c = np.array(c)  # Tobillo

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle


# Historial para seguimiento y tiempos
track_history = defaultdict(
    lambda: {"start_time": time.time(), "sitting_time": 0, "standing_time": 0, "last_pose": "unknown"})

# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================
rtsp_url = f"rtsp://{USUARIO}:{PASSWORD}@{NVR_IP}:554/cam/realmonitor?channel={CANAL}&subtype={SUBTYPE}"
stream = VideoCapturaHilada(rtsp_url)

# Configuración de Ventana Completa
window_name = "Jetson Orin - Dashboard de Deteccion IA"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, mouse_callback)

print("Sistema Iniciado.")
print("CONTROLES: [1] Línea | [2] Zona Análisis | [3] Zona Ley Silla | [R] Reset | [Q] Salir")

while True:
    frame = stream.read()
    if frame is None: continue

    # 1. INFERENCIA YOLO (Tracking activado)
    # persist=True mantiene el ID de las personas entre frames
    results = model.track(frame, persist=True, verbose=False, classes=[0])  # Clase 0 es persona

    # Copia para dibujar superposiciones (zonas transparentes)
    overlay = frame.copy()

    # --- DIBUJAR ZONAS ---
    # Líneas
    for l in zonas["lineas"]:
        cv2.line(frame, l[0], l[1], (0, 255, 255), 3)  # Amarillo

    # Zona Análisis (Azul transparente)
    for poly in zonas["analisis"]:
        cv2.fillPoly(overlay, [poly], (255, 0, 0))
        cv2.polylines(frame, [poly], True, (255, 0, 0), 2)

    # Zona Ergonomía / Ley Silla (Verde transparente)
    for poly in zonas["ergonomia"]:
        cv2.fillPoly(overlay, [poly], (0, 255, 0))
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

    # Aplicar transparencia
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # --- PROCESAMIENTO DE DETECCIONES ---
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        keypoints = results[0].keypoints.xy.cpu().numpy()  # Esqueleto

        for box, track_id, kpts in zip(boxes, track_ids, keypoints):
            x1, y1, x2, y2 = map(int, box)
            centro_x, centro_y = int((x1 + x2) / 2), int(y2)  # Punto en los pies

            # Verificar en qué zona está la persona
            en_zona_analisis = False
            en_zona_ergo = False

            # Checar Zona Análisis
            for poly in zonas["analisis"]:
                if cv2.pointPolygonTest(poly, (centro_x, centro_y), False) >= 0:
                    en_zona_analisis = True
                    # Aquí podrías dibujar trayectorias
                    cv2.circle(frame, (centro_x, centro_y), 5, (255, 0, 0), -1)

            # Checar Zona Ergonomía (Ley Silla)
            for poly in zonas["ergonomia"]:
                if cv2.pointPolygonTest(poly, (centro_x, centro_y), False) >= 0:
                    en_zona_ergo = True

            # LÓGICA LEY SILLA (Solo si está en zona ergo o si no hay zonas definidas aun)
            if en_zona_ergo or (not zonas["ergonomia"] and not zonas["analisis"]):
                # Puntos clave del esqueleto (COCO Keypoints):
                # 11: Cadera Izq, 13: Rodilla Izq, 15: Tobillo Izq
                # 12: Cadera Der, 14: Rodilla Der, 16: Tobillo Der

                # Usamos la pierna derecha como referencia (puedes promediar ambas)
                hip, knee, ankle = kpts[12], kpts[14], kpts[16]

                # Si la confianza de los puntos es baja (0,0), ignoramos
                if np.sum(hip) > 0 and np.sum(knee) > 0:
                    angulo_rodilla = calcular_angulo(hip, knee, ankle)

                    # Umbral: Si la rodilla está doblada < 140 grados, está sentado
                    estado = "Sentado" if angulo_rodilla < 140 else "Parado"
                    color_status = (0, 0, 255) if estado == "Sentado" else (0, 255, 0)

                    # Actualizar contadores
                    if estado == "Sentado":
                        track_history[track_id]["sitting_time"] += 1 / 30  # Asumiendo 30fps aprox
                    else:
                        track_history[track_id]["standing_time"] += 1 / 30

                    # Visualización
                    cv2.putText(frame, f"{estado} ({int(angulo_rodilla)} deg)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)

                    # Mostrar tiempos acumulados sobre la cabeza
                    t_sit = int(track_history[track_id]["sitting_time"])
                    t_stand = int(track_history[track_id]["standing_time"])
                    cv2.putText(frame, f"Silla: {t_sit}s | Pie: {t_stand}s", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Dibujar Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # --- GUI INSTRUCCIONES ---
    cv2.putText(frame, "MODO ACTUAL: " + (modo_dibujo if modo_dibujo else "Navegacion"), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Dibujar puntos temporales si se está creando una zona
    if len(puntos_temp) > 0:
        for pt in puntos_temp:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset zonas
        zonas = {"lineas": [], "analisis": [], "ergonomia": []}
        print("Zonas borradas.")
    elif key == ord('1'):
        modo_dibujo = "linea"; puntos_temp = []; print("Modo Linea")
    elif key == ord('2'):
        modo_dibujo = "analisis"; puntos_temp = []; print("Modo Analisis")
    elif key == ord('3'):
        modo_dibujo = "ergonomia"; puntos_temp = []; print("Modo Ley Silla")

stream.release()
cv2.destroyAllWindows()