import cv2
import threading
import queue
import time
import numpy as np

# ==========================================
# --- CONFIGURACIÓN ---
# ==========================================
NVR_IP = "192.168.1.108"
USUARIO = "admin"
PASSWORD = "admin2025"
CANAL = "2"
SUBTYPE = "0"

# --- AJUSTES DE VISUALIZACIÓN DE PUNTOS ---
# GRID_STEP: Define qué tan densa es la malla.
# Un valor bajo (ej. 5) = muchos puntos juntos (más lento).
# Un valor alto (ej. 20) = puntos más separados (más rápido).
GRID_STEP = 12

# DOT_RADIUS: El tamaño del punto verde.
DOT_RADIUS = 2

# NOTA SOBRE AREA_MINIMA: Con la visualización de puntos, esta variable
# ya no se usa para filtrar contornos. Si ves mucho "ruido" de puntos
# verdes, ajusta el 'varThreshold' en la creación del sustractor MOG2 más abajo.


# ==========================================
# CLASE PARA LECTURA DE VIDEO SIN DELAY
# ==========================================
class VideoCapturaHilada:
    """Lee frames en hilo separado para evitar lag."""
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
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
# PROGRAMA PRINCIPAL
# ==========================================

rtsp_url = f"rtsp://{USUARIO}:{PASSWORD}@{NVR_IP}:554/cam/realmonitor?channel={CANAL}&subtype={SUBTYPE}"
print(f"Conectando a: {rtsp_url} ...")
stream_thread = VideoCapturaHilada(rtsp_url)

time.sleep(1.0)

# Inicializar sustractor.
# Ajusta 'varThreshold' (ej. de 25 a 50) si ves demasiados puntos por ruido.
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

print("Iniciando procesamiento con malla de puntos. Presiona 'q' para salir.")

while True:
    try:
        frame = stream_thread.read()
    except queue.Empty:
        continue

    if frame is None:
        break

    # --- PROCESAMIENTO ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Obtener máscara de movimiento
    fgmask = fgbg.apply(blurred)

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # Umbralizar para asegurar blanco/negro puro (la máscara final)
    _, thresh = cv2.threshold(fgmask, 240, 255, cv2.THRESH_BINARY)

    # ==================================================================
    # D. NUEVA VISUALIZACIÓN: MALLA DE PUNTOS (GRID POINTS)
    # ==================================================================
    height, width = thresh.shape
    motion_detected = False

    # Iteramos sobre la imagen usando saltos definidos por GRID_STEP
    # Esto crea la "malla" virtual.
    for y in range(0, height, GRID_STEP):
        for x in range(0, width, GRID_STEP):
            # Verificamos el valor del pixel en la máscara binaria 'thresh'.
            # Si es 255 (blanco), significa que el algoritmo detectó movimiento ahí.
            if thresh[y, x] == 255:
                motion_detected = True
                # Dibujamos un círculo verde relleno en el frame original
                # Coordenadas (x, y), radio, color BGR (verde), grosor -1 (relleno)
                cv2.circle(frame, (x, y), DOT_RADIUS, (0, 255, 0), -1)

    # E. Mostrar estado
    status_text = "Estado: Movimiento DETECTADO" if motion_detected else "Estado: Inactivo"
    color_text = (0, 0, 255) if motion_detected else (0, 255, 0)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)

    # F. Mostrar resultados
    cv2.imshow("Saxxon - Malla de Movimiento", frame)
    # Opcional: ver la máscara cruda
    # cv2.imshow("Mascara Thresh", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream_thread.release()
cv2.destroyAllWindows()
print("Programa finalizado.")