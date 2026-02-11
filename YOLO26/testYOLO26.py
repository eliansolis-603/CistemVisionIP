import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

# --- CONFIGURACIÓN ---
# CAMBIO PRINCIPAL: Usamos la versión 'small' de YOLO26
# Si no tienes el peso descargado, la librería intentará bajarlo automáticamente.
MODEL_NAME = "yolo11s.pt"

# URL RTSP (Asegúrate de que tus credenciales e IP sean correctas)
RTSP_URL = "rtsp://admin123:admin123@192.168.1.214:554/stream1"


# --- CLASE PARA LECTURA SIN LAG (BUFFERLESS) ---
class CameraStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        # Optimizamos el buffer interno de OpenCV si es posible
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
                time.sleep(0.1)

    def read(self):
        return self.ret, self.frame if self.ret else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# --- VARIABLES GLOBALES ---
puntos_roi = []


def dibujar_roi(event, x, y, flags, param):
    global puntos_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos_roi.append((x, y))


def es_punto_en_poligono(punto, poligono):
    """
    Verifica si un punto (x, y) está dentro del polígono.
    Devuelve True si está dentro o en el borde.
    """
    test = cv2.pointPolygonTest(poligono, punto, False)
    return test >= 0


def main():
    print(f"--- INICIALIZANDO YOLO26 ({MODEL_NAME}) ---")
    # Al ser YOLO26 (NMS-free), la carga puede ser ligeramente más rápida
    model = YOLO(MODEL_NAME)

    print(f"Conectando a {RTSP_URL}...")
    cam = CameraStream(RTSP_URL).start()

    time.sleep(1.0)  # Esperar a que el buffer de la cámara llene el primer frame
    ret, frame = cam.read()
    if frame is None:
        print("Error al conectar. Revisa IP/Credenciales.")
        cam.stop()
        return

    # --- FASE 1: SELECCIONAR ROI ---
    cv2.namedWindow("Configuracion - YOLO26")
    cv2.setMouseCallback("Configuracion - YOLO26", dibujar_roi)

    roi_poly = None

    print("--- DIBUJA EL AREA ---")
    print("Click Izquierdo: Poner punto")
    print("Enter: Confirmar y cerrar figura")

    while True:
        ret, current_frame = cam.read()
        if not ret: continue

        display_frame = current_frame.copy()

        if len(puntos_roi) > 0:
            pts = np.array(puntos_roi, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], False, (0, 255, 255), 2)
            for p in puntos_roi:
                cv2.circle(display_frame, p, 4, (0, 0, 255), -1)

        cv2.imshow("Configuracion - YOLO26", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(puntos_roi) > 2:  # Enter
            roi_poly = np.array([puntos_roi], dtype=np.int32)
            break
        elif key == ord('q'):
            cam.stop()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Configuracion - YOLO26")

    # --- FASE 2: BUCLE PRINCIPAL ---
    print("Iniciando vigilancia con Arquitectura NMS-Free...")

    while True:
        ret, frame = cam.read()
        if not ret: break

        start_time = time.time()

        # 1. Inferencia YOLO26
        # classes=[0] sigue siendo 'person' en el dataset COCO estándar
        # Nota: YOLO26 suele ser más robusto con la confianza, puedes probar subir conf=0.6 si ves falsos positivos
        results = model(frame, classes=[0], imgsz=640, verbose=False, conf=0.5)

        final_frame = frame.copy()

        # 2. Dibujar ROI
        overlay = final_frame.copy()
        cv2.fillPoly(overlay, [roi_poly], (255, 200, 100))
        cv2.addWeighted(overlay, 0.2, final_frame, 0.8, 0, final_frame)
        cv2.polylines(final_frame, [roi_poly], True, (255, 200, 100), 2)

        # 3. Procesar detecciones
        personas_en_area = 0

        # La estructura de salida de YOLO26 en ultralytics se mantiene compatible con v8/v11
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Usamos el punto medio inferior (pies) para mejor lógica de "entrar/salir"
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)

                esta_dentro = es_punto_en_poligono((center_x, center_y), roi_poly)

                if esta_dentro:
                    personas_en_area += 1
                    color = (0, 255, 0)
                    texto_estado = "DENTRO"
                else:
                    color = (0, 0, 255)
                    texto_estado = ""

                # Visualización
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(final_frame, (center_x, center_y), 5, color, -1)

        # 4. Info en pantalla
        cv2.putText(final_frame, f"PERSONAS EN AREA: {personas_en_area}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(final_frame, f"MODELO: YOLO26s", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(final_frame, f"FPS: {fps:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Vigilancia YOLO26", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()