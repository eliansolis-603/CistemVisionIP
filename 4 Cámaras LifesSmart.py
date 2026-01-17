import cv2
import threading
import time
from ultralytics import YOLO

# Lista de URLs (Reemplaza con las que descubras con ONVIF Device Manager)
# Si usas la Estación, podrían ser canales diferentes de la misma IP.
CAMERA_URLS = [
    'rtsp://admin:pass@192.168.1.62:554/live/ch0',  # Cámara 1
    'rtsp://admin:pass@192.168.1.63:554/live/ch0',  # Cámara 2
    'rtsp://admin:pass@192.168.1.64:554/live/ch0',  # Cámara 3
    'rtsp://admin:pass@192.168.1.65:554/live/ch0'  # Cámara 4
]


class CameraStream:
    """Clase para leer el flujo de video en un hilo separado (sin lag)"""

    def __init__(self, url, cam_id):
        self.url = url
        self.id = cam_id
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        # Iniciar hilo de lectura
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    # Reintento automático si se cae la conexión
                    print(f"⚠️ Cam {self.id}: Reconectando...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.url)
            else:
                time.sleep(0.5)

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


def main():
    # 1. Cargar modelo YOLO (una sola vez para todos)
    print("Cargando modelo YOLO...")
    model = YOLO('yolov8n.pt')

    # 2. Iniciar streams en paralelo
    streams = []
    for i, url in enumerate(CAMERA_URLS):
        print(f"Iniciando cámara {i + 1}...")
        streams.append(CameraStream(url, i + 1))

    time.sleep(2)  # Dar tiempo para conectar buffer

    print("✅ Sistema de 4 cámaras activo. Presiona 'q' para salir.")

    while True:
        # Procesar cada cámara
        for i, stream in enumerate(streams):
            frame = stream.get_frame()

            if frame is not None:
                # Opcional: Reducir tamaño para ganar velocidad en 4 cámaras
                # frame = cv2.resize(frame, (640, 360))

                # Detección
                results = model(frame, classes=[0], conf=0.5, verbose=False)
                annotated_frame = results[0].plot()

                # Mostrar en ventanas separadas (o podrías unirlas en una sola grid)
                cv2.imshow(f"Camara {i + 1}", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Limpieza
    for stream in streams:
        stream.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()