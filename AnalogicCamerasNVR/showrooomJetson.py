import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ==========================================
# --- CONFIGURACIÓN DEL NVR (VPN) ---
# ==========================================
NVR_IP = "192.168.1.112"  # IP Local del NVR (accesible por el túnel de la Mac)
USUARIO = "admin"
PASSWORD = "admin2025"  # <--- ¡Verifica que esta sea la contraseña correcta!
CANAL = "2"  # Número de cámara
SUBTYPE = "0"  # 0 = Main Stream (Alta Calidad), 1 = Sub Stream (Fluido)

# --- ESTÉTICA VISUAL ---
# Transparencia de las zonas (0.15 = muy sutil, como cristal)
ALPHA_ZONAS = 0.15

# Modelos
print("Cargando modelo YOLOv8 Pose en GPU...")
model = YOLO('yolov8n-pose.pt')


# ==========================================
# CLASE DE VIDEO HILADO (Optimizado para RTSP/VPN)
# ==========================================
class VideoCapturaHilada:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        # Nota: En RTSP, set() no suele cambiar la resolución real del NVR,
        # dependemos del 'subtype=0' para recibir HD.

        # Buffer pequeño para reducir latencia de red
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
                # Si se pierde la conexión VPN, espera antes de reintentar
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
# LÓGICA DE INTERFAZ Y DIBUJO
# ==========================================
puntos_temp = []
zonas = {"lineas": [], "analisis": [], "ergonomia": []}
modo_dibujo = None


def mouse_callback(event, x, y, flags, param):
    global puntos_temp, modo_dibujo
    if event == cv2.EVENT_LBUTTONDOWN:
        if modo_dibujo:
            puntos_temp.append((x, y))
            # Cerrar línea con 2 puntos
            if modo_dibujo == 'linea' and len(puntos_temp) == 2:
                zonas["lineas"].append(puntos_temp.copy())
                puntos_temp = []
                modo_dibujo = None
                print("Línea creada.")
            # Cerrar zona con 4 puntos
            elif modo_dibujo in ['analisis', 'ergonomia'] and len(puntos_temp) == 4:
                zonas[modo_dibujo].append(np.array(puntos_temp, np.int32))
                puntos_temp = []
                modo_dibujo = None
                print(f"Zona {modo_dibujo} guardada.")


def calcular_angulo(a, b, c):
    """Calcula el ángulo de la rodilla"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle


# Historial de tiempos por ID de persona
track_history = defaultdict(lambda: {"sitting_time": 0, "standing_time": 0})

# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================
# Construcción de la URL RTSP
rtsp_url = f"rtsp://{USUARIO}:{PASSWORD}@{NVR_IP}:554/cam/realmonitor?channel={CANAL}&subtype={SUBTYPE}"
print(f"Conectando al NVR en: {NVR_IP} (Canal {CANAL})...")

stream = VideoCapturaHilada(rtsp_url)

# Esperar un momento a que llegue el primer frame del túnel
time.sleep(2.0)

window_name = "Expo Vision Artificial - Jetson Orin"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# Activar pantalla completa para la exposición
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, mouse_callback)

print("Sistema Listo. Controles: [1] Línea | [2] Zona Análisis | [3] Zona Ley Silla | [R] Reset | [Q] Salir")

while True:
    frame = stream.read()
    if frame is None:
        # Si no llega video, mostramos mensaje en consola pero no rompemos el loop inmediatamente
        # print("Esperando flujo de video...")
        time.sleep(0.01)
        continue

    # 1. Inferencia YOLO (Tracking persistente)
    results = model.track(frame, persist=True, verbose=False, classes=[0], tracker="bytetrack.yaml")

    # 2. Capa de Visualización (Zonas Transparentes)
    overlay = frame.copy()

    # Dibujar Líneas (Amarillo)
    for l in zonas["lineas"]:
        cv2.line(frame, l[0], l[1], (0, 255, 255), 3)

    # Dibujar Zonas Rellenas en el Overlay
    for poly in zonas["analisis"]:  # Azul
        cv2.fillPoly(overlay, [poly], (255, 100, 0))
    for poly in zonas["ergonomia"]:  # Verde
        cv2.fillPoly(overlay, [poly], (0, 200, 0))

    # Aplicar transparencia (Mezcla el overlay con el frame original)
    cv2.addWeighted(overlay, ALPHA_ZONAS, frame, 1 - ALPHA_ZONAS, 0, frame)

    # Dibujar contornos de las zonas para dar definición
    for poly in zonas["analisis"]:
        cv2.polylines(frame, [poly], True, (255, 150, 0), 2)
    for poly in zonas["ergonomia"]:
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

    # 3. Procesamiento de Detecciones
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        keypoints = results[0].keypoints.xy.cpu().numpy()

        for box, track_id, kpts in zip(boxes, track_ids, keypoints):
            x1, y1, x2, y2 = map(int, box)
            centro_x, centro_y = int((x1 + x2) / 2), int(y2)

            # Verificar si está dentro de alguna zona
            en_zona_analisis = any(cv2.pointPolygonTest(p, (centro_x, centro_y), False) >= 0 for p in zonas["analisis"])
            en_zona_ergo = any(cv2.pointPolygonTest(p, (centro_x, centro_y), False) >= 0 for p in zonas["ergonomia"])

            # --- LÓGICA "LEY SILLA" ---
            # Se activa si está en zona verde O si no hay zonas definidas (modo libre)
            if en_zona_ergo or (not zonas["ergonomia"] and not zonas["analisis"]):
                # Índices COCO: 12=Cadera Der, 14=Rodilla Der, 16=Tobillo Der
                hip, knee, ankle = kpts[12], kpts[14], kpts[16]

                # Solo calculamos si la IA detectó la pierna con confianza
                if np.sum(hip) > 0 and np.sum(knee) > 0:
                    angulo = calcular_angulo(hip, knee, ankle)

                    # Umbral de postura
                    estado = "Sentado" if angulo < 140 else "De Pie"
                    color_status = (0, 50, 255) if estado == "Sentado" else (50, 255, 50)  # Rojo vs Verde

                    # Actualizar contadores
                    if estado == "Sentado":
                        track_history[track_id]["sitting_time"] += 1 / 30  # +1 frame (aprox 33ms)
                    else:
                        track_history[track_id]["standing_time"] += 1 / 30

                    # --- ETIQUETAS DE DATOS ---
                    # Fondo negro detrás del texto para legibilidad perfecta
                    label_estado = f"ID {track_id}: {estado}"
                    label_tiempos = f"Silla: {int(track_history[track_id]['sitting_time'])}s | Pie: {int(track_history[track_id]['standing_time'])}s"

                    # Dibujar fondo negro
                    cv2.rectangle(frame, (x1, y1 - 45), (x1 + 220, y1), (0, 0, 0), -1)
                    # Texto Estado
                    cv2.putText(frame, label_estado, (x1 + 5, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
                    # Texto Tiempos
                    cv2.putText(frame, label_tiempos, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1)

            # Dibujar Caja (Blanco Brillante para resaltar sobre las zonas tenues)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Indicador extra para zona de análisis
            if en_zona_analisis:
                cv2.circle(frame, (centro_x, centro_y), 8, (255, 100, 0), -1)  # Punto naranja
                cv2.circle(frame, (centro_x, centro_y), 10, (255, 255, 255), 2)  # Borde blanco

    # --- GUI INSTRUCCIONES ---
    modo_txt = f"MODO ACTIVO: {modo_dibujo.upper()}" if modo_dibujo else "NAVEGACION (Use teclas 1, 2, 3)"
    # Barra negra superior para instrucciones
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, modo_txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Dibujar puntos mientras se crea una zona
    if len(puntos_temp) > 0:
        for pt in puntos_temp:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.line(frame, pt, (pt[0] + 5, pt[1]), (0, 0, 255), 1)  # Pequeña cruz visual

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        zonas = {"lineas": [], "analisis": [], "ergonomia": []}; print("Zonas Reiniciadas")
    elif key == ord('1'):
        modo_dibujo = "linea"; puntos_temp = []
    elif key == ord('2'):
        modo_dibujo = "analisis"; puntos_temp = []
    elif key == ord('3'):
        modo_dibujo = "ergonomia"; puntos_temp = []

stream.release()
cv2.destroyAllWindows()