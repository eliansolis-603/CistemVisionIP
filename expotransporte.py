import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
from datetime import datetime
from collections import defaultdict
from threading import Thread
import time


# --- CLASE PARA ELIMINAR EL LAG (THREADING) ---
class CameraStream:
    def __init__(self, rtsp_url):
        self.stream = cv2.VideoCapture(rtsp_url)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()


# --- FUNCIÓN ESTÉTICA ---
def draw_rounded_rect(img, pt1, pt2, color, thickness, r):
    x1, y1 = pt1
    x2, y2 = pt2
    # Ajuste de radio seguro
    w_rect = abs(x2 - x1)
    h_rect = abs(y2 - y1)
    r = min(r, w_rect // 2, h_rect // 2)

    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# --- CONFIGURACIÓN ---
RTSP_URL = 'rtsp://admin123:admin123@192.168.1.187/stream1'
MODEL_PATH = 'yolov8n.pt'  # Asegúrate de tener este modelo descargado

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "TapoControl")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
CSV_SUMMARY = os.path.join(OUTPUT_DIR, "registro_zona.csv")

# --- INICIALIZACIÓN ---
print("Cargando modelo...")
model = YOLO(MODEL_PATH)
track_history = defaultdict(lambda: [])
people_records = {}  # {id: start_time}

# Crear CSV si no existe
if not os.path.exists(CSV_SUMMARY):
    with open(CSV_SUMMARY, 'w', newline='') as f:
        csv.writer(f).writerow(["ID", "Clase", "Fecha", "Hora_Entrada", "Estado", "Duracion_Actual"])

# --- PASO 1: SELECCIÓN DE ROI ---
print("Conectando cámara para configurar zona...")
cap_temp = cv2.VideoCapture(RTSP_URL)
ret, first_frame = cap_temp.read()
cap_temp.release()

if not ret:
    print("Error: No se pudo conectar a la cámara para definir ROI.")
    exit()

print("--- INSTRUCCIONES ---")
print("1. Dibuja un rectángulo con el mouse sobre el área a vigilar.")
print("2. Presiona ENTER o ESPACIO para confirmar.")
print("3. Presiona 'c' si quieres cancelar la selección.")

roi_box = cv2.selectROI("DEFINIR ZONA (Enter para confirmar)", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("DEFINIR ZONA (Enter para confirmar)")

# roi_box devuelve (x, y, w, h). Si w o h son 0, no se seleccionó nada.
roi_active = roi_box[2] > 0 and roi_box[3] > 0
if roi_active:
    rx, ry, rw, rh = int(roi_box[0]), int(roi_box[1]), int(roi_box[2]), int(roi_box[3])
    print(f"Zona definida: x={rx}, y={ry}, w={rw}, h={rh}")
else:
    print("No se seleccionó zona. Se monitoreará toda la pantalla (sin filtro de área).")
    rx, ry, rw, rh = 0, 0, first_frame.shape[1], first_frame.shape[0]

# --- PASO 2: STREAMING EN VIVO ---
print("Iniciando vigilancia...")
cam_stream = CameraStream(RTSP_URL).start()
time.sleep(1.0)

while True:
    frame = cam_stream.read()
    if frame is None: continue

    # Dibujar la ROI definida visualmente (Rectángulo rojo fino)
    if roi_active:
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
        cv2.putText(frame, "ZONA VIGILADA", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- TRACKING DE TODOS LOS OBJETOS ---
    # Eliminamos classes=0 para que detecte ODO
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", imgsz=640)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls_ids = results[0].boxes.cls.int().cpu().tolist()  # IDs de clase (0=persona, etc)

        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            x, y, w_box, h_box = box

            # Centroide del objeto
            cx, cy = int(x), int(y)

            # Coordenadas visuales
            tl = (int(x - w_box / 2), int(y - h_box / 2))
            br = (int(x + w_box / 2), int(y + h_box / 2))

            # Obtener nombre de la clase (ej: "person", "chair")
            class_name = model.names[cls_id]

            # --- DIBUJO VISUAL (PARA TODOS) ---
            color = (0, 255, 0) if cls_id == 0 else (200, 200, 200)  # Verde para personas, Gris para otros
            draw_rounded_rect(frame, tl, br, color, 2, 10)
            cv2.putText(frame, f"{class_name} {track_id}", (tl[0], tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # --- LÓGICA DE REGISTRO (SOLO PERSONAS EN ROI) ---
            # 1. ¿Es una persona? (Clase 0 en COCO es persona)
            # 2. ¿Está el centroide dentro de la ROI?
            is_person = (cls_id == 0)
            in_zone = (rx < cx < rx + rw) and (ry < cy < ry + rh)

            if is_person and in_zone:
                # Si entra por primera vez a la zona o al registro
                if track_id not in people_records:
                    people_records[track_id] = datetime.now()
                    # Opcional: Registrar entrada inmediata
                    print(f"Alerta: Persona {track_id} entró en zona.")

                # Calcular duración
                start_t = people_records[track_id]
                duration = (datetime.now() - start_t).total_seconds()

                # Actualizar texto en pantalla con tiempo
                cv2.putText(frame, f"{int(duration)}s en zona", (tl[0], tl[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

                # Guardar registro periódico (o al salir, aquí guardamos cada X frames para no saturar IO,
                # pero para este ejemplo, guardamos si el tiempo es múltiplo de 5 seg aprox para no llenar el CSV)
                if int(duration) > 0 and int(duration) % 5 == 0:
                    with open(CSV_SUMMARY, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            track_id,
                            class_name,
                            datetime.now().strftime("%Y-%m-%d"),
                            start_t.strftime("%H:%M:%S"),
                            "DENTRO_ZONA",
                            round(duration, 2)
                        ])

            # Si estaba registrado pero ya salió de la zona, podrías limpiar el registro
            # o marcar salida. Aquí lo mantenemos simple.

    cv2.imshow("Monitor ROI + Objetos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_stream.stop()
cv2.destroyAllWindows()