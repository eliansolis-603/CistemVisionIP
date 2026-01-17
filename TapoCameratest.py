import cv2
import threading
import queue
import time
import numpy as np

# ==========================================
# --- CONFIGURACIÓN (TUS DATOS) ---
# ==========================================
NVR_IP = "192.168.1.108"
USUARIO = "admin"
PASSWORD = "admin2025"
CANAL = "2"
SUBTYPE = "0"  # Alta definición

# --- SENSIBILIDAD ---
SENSITIVITY_THRESHOLD = 25  # Menos valor = más sensible
MIN_AREA_PIXELS = 500  # Umbral para el texto de "Movimiento Detectado"


# ==========================================
# CLASE PARA LECTURA DE VIDEO SIN DELAY (Indispensable para Starlink)
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
# PROGRAMA PRINCIPAL
# ==========================================

rtsp_url = f"rtsp://{USUARIO}:{PASSWORD}@{NVR_IP}:554/cam/realmonitor?channel={CANAL}&subtype={SUBTYPE}"
print(f"Conectando a: {rtsp_url} ...")

stream_thread = VideoCapturaHilada(rtsp_url)

# Esperamos el primer frame para inicializar la comparación
time.sleep(2.0)
frame_prev = stream_thread.read()
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)

print("Sistema iniciado. Presiona 'q' para salir.")

while True:
    # A. Obtener frame actual
    frame_curr = stream_thread.read()
    if frame_curr is None:
        break

    # B. Preprocesamiento
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.GaussianBlur(gray_curr, (21, 21), 0)

    # C. Calcular Diferencia Absoluta (Muy rápido)
    frame_diff = cv2.absdiff(gray_prev, gray_curr)

    # D. Aplicar Umbral y Dilatación
    # Esto crea la máscara de los "puntos" de movimiento
    _, mask = cv2.threshold(frame_diff, SENSITIVITY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Dilatamos para que los puntos sean más gruesos y notorios
    mask = cv2.dilate(mask, None, iterations=2)

    # E. REPRESENTACIÓN VISUAL (Vectorizada - Super rápida)
    # Coloreamos de VERDE puro todos los píxeles donde hay movimiento
    frame_curr[mask == 255] = [0, 255, 0]

    # F. Lógica de detección para el texto
    motion_pixels = np.count_nonzero(mask)
    if motion_pixels > MIN_AREA_PIXELS:
        status_text = "MOVIMIENTO DETECTADO"
        color_text = (0, 0, 255)  # Rojo
    else:
        status_text = "Sistema Activo"
        color_text = (0, 255, 0)  # Verde

    cv2.putText(frame_curr, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_text, 2)

    # G. Actualizar frame previo
    gray_prev = gray_curr

    # H. Redimensionar para que se vea bien en tu monitor
    # Esto soluciona el problema de "ventana chica"
    display_frame = cv2.resize(frame_curr, (1280, 720))
    cv2.imshow('Saxxon - Deteccion por Diferencia de Frames', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream_thread.release()
cv2.destroyAllWindows()