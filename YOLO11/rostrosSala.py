import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from collections import deque

# --- CONFIGURACIÓN ---
MODEL_NAME = "yolo11s.pt"  # Usamos 's' para buen balance o 'n' para velocidad extrema
RTSP_URL = "rtsp://admin123:admin123@192.168.1.228:554/stream1"

# Ajustes Visuales
MAX_TRAIL_LENGTH = 50  # Largo de la cola de trayectoria
FACE_DASHBOARD_WIDTH = 300  # Ancho del panel lateral
ROW_HEIGHT = 80  # Altura de cada fila de "persona" en el panel


# --- CLASE DE CÁMARA (Bufferless) ---
class CameraStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame if self.ret else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# --- GESTOR DE ZONAS ---
drawing_points = []


def draw_poly(event, x, y, flags, param):
    global drawing_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and drawing_points:
        drawing_points.pop()


def configurar_zona(frame):
    global drawing_points
    drawing_points = []
    window_name = "CONFIGURACION: DIBUJA LA ZONA DE LA SALA"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_poly)

    print("--- INSTRUCCIONES ---")
    print("Click Izq: Marcar puntos del perímetro de la sala")
    print("Enter: Confirmar y empezar")

    poly = None
    while True:
        disp = frame.copy()
        cv2.putText(disp, "DIBUJA EL AREA DE LA SALA", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(drawing_points) > 0:
            pts = np.array(drawing_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], False, (0, 255, 255), 2)
            for p in drawing_points:
                cv2.circle(disp, p, 4, (0, 0, 255), -1)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(drawing_points) > 2:
            poly = np.array(drawing_points, np.int32)
            break
        elif key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    return poly


# --- CLASE VISITOR (Gestión de Identidad) ---
class VisitorTrack:
    def __init__(self, track_id, crop_img, start_time):
        self.id = track_id
        self.face_crop = crop_img
        self.start_time = start_time
        self.trail = deque(maxlen=MAX_TRAIL_LENGTH)  # Guarda historial de puntos
        self.last_seen_time = time.time()
        self.active = True


def main():
    print(f"Cargando modelo {MODEL_NAME}...")
    # Usamos track=True en YOLO, así que necesitamos cargar modelo normal
    model = YOLO(MODEL_NAME)

    print(f"Conectando cámara: {RTSP_URL}")
    cam = CameraStream(RTSP_URL).start()
    time.sleep(1.5)  # Buffer fill

    ret, setup_frame = cam.read()
    if setup_frame is None:
        print("Error de conexión.")
        cam.stop()
        return

    # 1. Definir la Sala
    zona_sala = configurar_zona(setup_frame)
    if zona_sala is None:
        cam.stop()
        return

    # Diccionario para guardar visitantes: { track_id: VisitorTrack }
    visitors_db = {}

    print("--- INICIANDO REGISTRO DE VISITANTES ---")

    while True:
        ret, frame = cam.read()
        if not ret: break

        current_time = time.time()

        # Preparar canvas expandido para el menú lateral
        h, w = frame.shape[:2]
        # Creamos una imagen más ancha para poner el menú a la derecha
        dashboard_w = FACE_DASHBOARD_WIDTH
        final_canvas = np.zeros((h, w + dashboard_w, 3), dtype=np.uint8)
        # Copiamos el video original a la parte izquierda
        final_canvas[0:h, 0:w] = frame

        # Fondo del panel derecho
        final_canvas[:, w:] = (30, 30, 30)  # Gris oscuro
        cv2.putText(final_canvas, "REGISTRO SALA", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. TRACKING CON YOLO
        # persist=True es VITAL para mantener el ID del sujeto frame a frame
        results = model.track(frame, classes=[0], persist=True, verbose=False, imgsz=640, tracker="bytetrack.yaml")

        active_ids_now = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Centro de los pies (para saber si está en sala)
                cx_feet, cy_feet = int((x1 + x2) / 2), int(y2)

                # Verificar si está en zona
                en_sala = cv2.pointPolygonTest(zona_sala, (cx_feet, cy_feet), False) >= 0

                if en_sala:
                    active_ids_now.append(track_id)

                    # --- GESTIÓN DEL VISITANTE ---
                    if track_id not in visitors_db:
                        # NUEVO VISITANTE: Registrar
                        # Cortamos la parte superior (aprox 20%) para simular "Cara"
                        # Si quieres cara real, se requiere modelo específico, pero esto es muy efectivo para overview
                        face_h = int((y2 - y1) * 0.25)
                        face_crop = frame[max(0, y1):max(0, y1 + face_h), max(0, x1):max(0, x2)]

                        # Resize para el dashboard
                        if face_crop.size > 0:
                            face_crop = cv2.resize(face_crop, (60, 60))
                        else:
                            face_crop = np.zeros((60, 60, 3), dtype=np.uint8)

                        visitors_db[track_id] = VisitorTrack(track_id, face_crop, current_time)

                    # ACTUALIZAR DATOS (Trayectoria)
                    visitor = visitors_db[track_id]
                    visitor.last_seen_time = current_time
                    visitor.active = True

                    # Agregar punto a trayectoria (centro del cuerpo)
                    cx_body, cy_body = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    visitor.trail.append((cx_body, cy_body))

                    # --- DIBUJAR EN VIDEO (Izquierda) ---
                    # Caja
                    cv2.rectangle(final_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # ID
                    cv2.putText(final_canvas, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)

                    # Dibujar Trayectoria
                    if len(visitor.trail) > 1:
                        pts = np.array(list(visitor.trail), np.int32).reshape((-1, 1, 2))
                        cv2.polylines(final_canvas, [pts], False, (0, 150, 255), 2)  # Naranja

        # --- RENDERIZAR DASHBOARD (Derecha) ---
        # Filtramos usuarios que han sido vistos recientemente (para no borrar del menu inmediatamente si parpadean)
        y_offset = 80

        # Ordenar: Los más recientes arriba
        active_visitors = [v for k, v in visitors_db.items() if (current_time - v.last_seen_time) < 2.0]

        for v in active_visitors:
            if y_offset + ROW_HEIGHT > h: break  # No cabe en pantalla

            # Calcular tiempo
            dwell_time = current_time - v.start_time
            mins = int(dwell_time // 60)
            secs = int(dwell_time % 60)

            # 1. Pegar la "Foto" (Face Crop)
            fx, fy = w + 20, y_offset
            face_h, face_w = v.face_crop.shape[:2]
            final_canvas[fy:fy + face_h, fx:fx + face_w] = v.face_crop

            # 2. Texto Info
            cv2.putText(final_canvas, f"ID: {v.id}", (fx + 70, fy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        1)
            cv2.putText(final_canvas, f"Time: {mins:02}:{secs:02}", (fx + 70, fy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1)

            # Línea separadora
            cv2.line(final_canvas, (w, y_offset + ROW_HEIGHT - 10), (w + dashboard_w, y_offset + ROW_HEIGHT - 10),
                     (100, 100, 100), 1)

            y_offset += ROW_HEIGHT

        # Dibujar zona en video
        cv2.polylines(final_canvas, [zona_sala], True, (255, 0, 0), 2)

        # Info general
        cv2.putText(final_canvas, f"Personas: {len(active_visitors)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("Monitor de Sala - Tracking & Dashboard", final_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()