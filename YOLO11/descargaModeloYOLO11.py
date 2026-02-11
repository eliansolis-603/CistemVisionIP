import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

# --- CONFIGURACIÓN ---
# Usamos YOLO11s como pediste
MODEL_NAME = "yolo11s.pt"

# URL RTSP (Nota: Si yolo11s va lento, cambia stream1 por stream2)
RTSP_URL = "rtsp://admin123:admin123@192.168.1.214:554/stream1"


# --- CLASE PARA LECTURA SIN LAG ---
class CameraStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
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
    # measureDist=False devuelve +1 si está dentro, -1 fuera, 0 en borde
    test = cv2.pointPolygonTest(poligono, punto, False)
    return test >= 0


def main():
    print(f"Cargando modelo {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print(f"Conectando a {RTSP_URL}...")
    cam = CameraStream(RTSP_URL).start()

    # Esperar buffer
    time.sleep(1.0)
    ret, frame = cam.read()
    if frame is None:
        print("Error al conectar. Revisa IP/Credenciales.")
        cam.stop()
        return

    # --- FASE 1: SELECCIONAR ROI ---
    cv2.namedWindow("Configuracion")
    cv2.setMouseCallback("Configuracion", dibujar_roi)

    roi_poly = None

    print("--- DIBUJA EL AREA ---")
    print("Click Izquierdo: Poner punto")
    print("Enter: Confirmar y cerrar figura")

    while True:
        ret, current_frame = cam.read()
        if not ret: continue

        display_frame = current_frame.copy()

        # Dibujar líneas del polígono mientras se crea
        if len(puntos_roi) > 0:
            pts = np.array(puntos_roi, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], False, (0, 255, 255), 2)
            for p in puntos_roi:
                cv2.circle(display_frame, p, 4, (0, 0, 255), -1)

        cv2.imshow("Configuracion", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(puntos_roi) > 2:  # Enter
            roi_poly = np.array([puntos_roi], dtype=np.int32)
            break
        elif key == ord('q'):
            cam.stop()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Configuracion")

    # --- FASE 2: BUCLE PRINCIPAL ---
    print("Iniciando vigilancia...")

    while True:
        ret, frame = cam.read()
        if not ret: break

        start_time = time.time()

        # 1. Inferencia sobre ODO el frame (sin máscara)
        #    imgsz=640 ayuda a mantener FPS altos aunque la cámara sea 1080p
        results = model(frame, classes=[0], imgsz=640, verbose=False, conf=0.5)

        # Copia para dibujar
        final_frame = frame.copy()

        # 2. Dibujar ROI (Sombreado tenue como pediste)
        overlay = final_frame.copy()
        cv2.fillPoly(overlay, [roi_poly], (255, 200, 100))  # Color Cyan
        cv2.addWeighted(overlay, 0.2, final_frame, 0.8, 0, final_frame)  # 20% opacidad
        cv2.polylines(final_frame, [roi_poly], True, (255, 200, 100), 2)

        # 3. Procesar detecciones y contar
        personas_en_area = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Coordenadas de la caja
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculamos el punto central de los pies (más preciso para ubicación en suelo)
                # O puedes usar el centro exacto: center_x, center_y
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)  # Pies
                # center_y = int((y1 + y2) / 2) # Centro del cuerpo (descomenta si prefieres esto)

                # Verificar si el punto está dentro del ROI
                esta_dentro = es_punto_en_poligono((center_x, center_y), roi_poly)

                if esta_dentro:
                    personas_en_area += 1
                    color = (0, 255, 0)  # Verde
                    texto_estado = "DENTRO"
                else:
                    color = (0, 0, 255)  # Rojo
                    texto_estado = ""

                # Dibujar caja y etiqueta
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(final_frame, (center_x, center_y), 5, color, -1)  # Punto de referencia

                # Etiqueta (opcional)
                # cv2.putText(final_frame, texto_estado, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 4. Mostrar información en pantalla
        cv2.putText(final_frame, f"PERSONAS EN AREA: {personas_en_area}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Calculo de FPS para monitoreo
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(final_frame, f"FPS: {fps:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Vigilancia YOLO11s", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()