import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import threading
import time
from collections import deque

# --- CONFIGURACIÓN ---
MODEL_NAME = "yolo11s.pt"
# Asegúrate de que admin123 sea la contraseña de la CUENTA DE CÁMARA (App Tapo -> Ajustes -> Avanzado -> Cuenta de cámara)
RTSP_URL = "rtsp://admin123:admin123@192.168.1.228:554/stream1"

# --- AJUSTE OBLIGATORIO PARA TAPO C310 ---
# Aunque la cámara sea de 3MP (2304x1296), forzamos a FFmpeg a que nos entregue 1080p
# para que coincida con el buffer de Python. NO CAMBIES ESTO.
WIDTH = 1920
HEIGHT = 1080

# Ajustes Visuales
MAX_TRAIL_LENGTH = 50
FACE_DASHBOARD_WIDTH = 300
ROW_HEIGHT = 80


# --- CLASE DE CÁMARA FFMPEG (REDIMENSIONADO AUTOMÁTICO) ---
class FFmpegStream:
    def __init__(self, url, width, height):
        self.url = url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()

        print(f"   > Iniciando FFmpeg forzando salida a {width}x{height}...")

        # Comando FFmpeg BLINDADO
        self.command = [
            'ffmpeg',
            '-y',  # Sobrescribir sin preguntar
            '-loglevel', 'error',  # Solo mostrar errores graves
            '-rtsp_transport', 'tcp',  # TCP para evitar cortes WiFi
            '-i', self.url,  # Entrada
            '-s', f'{width}x{height}',  # <--- CLAVE: Redimensionar a 1920x1080
            '-f', 'image2pipe',  # Salida a tubería
            '-pix_fmt', 'bgr24',  # Formato OpenCV
            '-vcodec', 'rawvideo',  # Sin compresión
            '-tune', 'zerolatency',  # Latencia cero
            '-'  # Salida a Python
        ]

        try:
            # Buffer grande para evitar cuellos de botella en el sistema operativo
            self.pipe = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 ** 8
            )
            self.running = True
        except Exception as e:
            print(f"Error crítico lanzando FFmpeg: {e}")
            self.running = False

        # Hilo de lectura en segundo plano
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        """Hilo voraz: Lee frames tan rápido como puede y guarda solo el último."""
        while self.running:
            # Leer tamaño EXACTO de un frame
            raw_image = self.pipe.stdout.read(self.frame_size)

            if len(raw_image) != self.frame_size:
                # Si los bytes no coinciden, algo malo pasó
                error_log = self.pipe.stderr.read(1024)
                if len(raw_image) == 0:
                    print(
                        f"\n[FFmpeg Error] No llegan datos. Posible error de contraseña/IP:\n{error_log.decode('utf-8', errors='ignore')}")
                    self.running = False
                    break
            else:
                # Convertir bytes a imagen
                image = np.frombuffer(raw_image, dtype='uint8')
                frame = image.reshape((self.height, self.width, 3))

                with self.lock:
                    self.latest_frame = frame

    def read(self):
        """Devuelve una copia del frame más reciente."""
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            else:
                return False, None

    def stop(self):
        self.running = False
        if self.pipe:
            self.pipe.terminate()
            try:
                self.pipe.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.pipe.kill()


# --- GESTOR DE ZONAS ---
drawing_points = []


def draw_poly(event, x, y, flags, param):
    global drawing_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and drawing_points:
        drawing_points.pop()


def configurar_zona(cam_stream):
    global drawing_points
    drawing_points = []
    window_name = "CONFIG: DIBUJA ZONA (Enter=Confirmar)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_poly)

    print("--- INSTRUCCIONES ---")
    print("Click Izq: Puntos. Enter: Terminar.")

    poly = None

    while True:
        ret, frame = cam_stream.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        # Redimensionar visualmente para que quepa en pantalla Mac (1080p es muy grande)
        display_h, display_w = 720, 1280
        frame_resized = cv2.resize(frame, (display_w, display_h))

        # Escalas para traducir clicks
        scale_x = frame.shape[1] / display_w
        scale_y = frame.shape[0] / display_h

        cv2.putText(frame_resized, "DIBUJA ZONA", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(drawing_points) > 0:
            pts_display = np.array(drawing_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_resized, [pts_display], False, (0, 255, 255), 2)
            for p in drawing_points:
                cv2.circle(frame_resized, p, 4, (0, 0, 255), -1)

        cv2.imshow(window_name, frame_resized)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(drawing_points) > 2:  # Enter
            real_points = []
            for (px, py) in drawing_points:
                real_points.append((int(px * scale_x), int(py * scale_y)))
            poly = np.array(real_points, np.int32)
            break
        elif key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    return poly


# --- CLASE VISITOR ---
class VisitorTrack:
    def __init__(self, track_id, crop_img, start_time):
        self.id = track_id
        self.face_crop = crop_img
        self.start_time = start_time
        self.trail = deque(maxlen=MAX_TRAIL_LENGTH)
        self.last_seen_time = time.time()
        self.active = True


# --- MAIN ---
def main():
    print(f"Cargando YOLO {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print(f"Conectando a Tapo C310 via FFmpeg (Forzando 1080p)...")
    cam = FFmpegStream(RTSP_URL, WIDTH, HEIGHT).start()

    print("Esperando primer frame (puede tardar 5-10 seg)...")
    frame_ok = False
    for i in range(100):  # 10 segundos de timeout
        ret, _ = cam.read()
        if ret:
            frame_ok = True
            break
        time.sleep(0.1)

    if not frame_ok:
        print("ERROR: No se recibió video. Revisa consola por errores de FFmpeg.")
        cam.stop()
        return
    else:
        print(">> VIDEO RECIBIDO CORRECTAMENTE.")

    # 1. Definir la Sala
    zona_sala = configurar_zona(cam)
    if zona_sala is None:
        cam.stop()
        return

    visitors_db = {}
    print("--- VIGILANCIA ACTIVA (SIN LAG) ---")

    while True:
        # Lectura instantánea
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.01)
            continue

        current_time = time.time()
        h, w = frame.shape[:2]

        # Dashboard Canvas
        dashboard_w = FACE_DASHBOARD_WIDTH
        final_canvas = np.zeros((h, w + dashboard_w, 3), dtype=np.uint8)
        final_canvas[0:h, 0:w] = frame
        final_canvas[:, w:] = (30, 30, 30)
        cv2.putText(final_canvas, "REGISTRO SALA", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. YOLO TRACKING
        results = model.track(frame, classes=[0], persist=True, verbose=False, imgsz=640, tracker="bytetrack.yaml")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx_feet, cy_feet = int((x1 + x2) / 2), int(y2)

                en_sala = cv2.pointPolygonTest(zona_sala, (cx_feet, cy_feet), False) >= 0

                if en_sala:
                    if track_id not in visitors_db:
                        # Crop Cara Simulado
                        face_h = int((y2 - y1) * 0.25)
                        face_crop = frame[max(0, y1):max(0, y1 + face_h), max(0, x1):max(0, x2)]

                        if face_crop.size > 0:
                            face_crop = cv2.resize(face_crop, (60, 60))
                        else:
                            face_crop = np.zeros((60, 60, 3), dtype=np.uint8)

                        visitors_db[track_id] = VisitorTrack(track_id, face_crop, current_time)

                    visitor = visitors_db[track_id]
                    visitor.last_seen_time = current_time
                    visitor.trail.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

                    # Dibujar
                    cv2.rectangle(final_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(final_canvas, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

                    if len(visitor.trail) > 1:
                        pts = np.array(list(visitor.trail), np.int32).reshape((-1, 1, 2))
                        cv2.polylines(final_canvas, [pts], False, (0, 150, 255), 2)

        # 3. RENDER DASHBOARD
        y_offset = 80
        active_visitors = [v for k, v in visitors_db.items() if (current_time - v.last_seen_time) < 3.0]
        active_visitors.sort(key=lambda x: x.id)

        for v in active_visitors:
            if y_offset + ROW_HEIGHT > h: break

            dwell_time = current_time - v.start_time
            mins, secs = int(dwell_time // 60), int(dwell_time % 60)

            fx, fy = w + 20, y_offset
            face_h, face_w = v.face_crop.shape[:2]
            final_canvas[fy:fy + face_h, fx:fx + face_w] = v.face_crop

            cv2.putText(final_canvas, f"ID: {v.id}", (fx + 70, fy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        1)
            cv2.putText(final_canvas, f"{mins:02}:{secs:02}", (fx + 70, fy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1)

            cv2.line(final_canvas, (w, y_offset + ROW_HEIGHT - 10), (w + dashboard_w, y_offset + ROW_HEIGHT - 10),
                     (100, 100, 100), 1)
            y_offset += ROW_HEIGHT

        # Zona
        cv2.polylines(final_canvas, [zona_sala], True, (255, 0, 0), 2)

        # Redimensionar salida a 720p para ver en pantalla Mac
        display_frame = cv2.resize(final_canvas, (1280, 720))
        cv2.imshow("Monitor Sala (Tapo C310 Fixed)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()