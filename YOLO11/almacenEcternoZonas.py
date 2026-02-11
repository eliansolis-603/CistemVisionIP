import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

# --- CONFIGURACIÓN ---
MODEL_NAME = "yolo11s.pt"
# Ajusta aquí tu IP y credenciales
RTSP_URL = "rtsp://admin123:admin123@192.168.1.78:554/stream1"

# --- COLORES Y ESTILOS (Lógica de Almacén) ---
COLOR_BAHIA_LIBRE = (0, 255, 0)  # Verde
COLOR_BAHIA_OCUPADA = (0, 140, 255)  # Naranja
COLOR_PUERTA_SAFE = (255, 255, 0)  # Cyan/Amarillo
COLOR_PUERTA_RISK = (0, 0, 255)  # Rojo
COLOR_KPI_BG = (50, 50, 50)  # Fondo del panel


# --- CLASE PARA LECTURA SIN LAG (BUFFERLESS) ---
class CameraStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        # Forzamos buffer pequeño para minimizar latencia
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


# --- GESTIÓN DE ZONAS (MOUSE) ---
drawing_points = []


def draw_polygon_callback(event, x, y, flags, param):
    global drawing_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if drawing_points:
            drawing_points.pop()


def get_zone_on_static_frame(frame, window_name, instruction_text):
    """
    Permite dibujar zonas sobre una imagen estática capturada del stream.
    """
    global drawing_points
    drawing_points = []  # Resetear puntos

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon_callback)

    print(f"--- CONFIGURANDO: {window_name} ---")

    while True:
        img_copy = frame.copy()

        # Instrucciones visuales
        cv2.rectangle(img_copy, (0, 0), (900, 80), (0, 0, 0), -1)
        cv2.putText(img_copy, instruction_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_copy, "Click Izq: Punto | Click Der: Borrar | ENTER: Confirmar", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Dibujar polígono en construcción
        if len(drawing_points) > 0:
            pts = np.array(drawing_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts], False, (0, 255, 255), 2)
            for pt in drawing_points:
                cv2.circle(img_copy, pt, 4, (0, 0, 255), -1)

        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(1) & 0xFF

        # Enter para confirmar
        if key == 13 and len(drawing_points) > 2:
            break
        # 'q' para salir de emergencia
        elif key == ord('q'):
            return None

    cv2.destroyWindow(window_name)
    return np.array(drawing_points, np.int32)


def is_inside(center, polygon):
    """Devuelve True si el punto está dentro del polígono"""
    return cv2.pointPolygonTest(polygon, center, False) >= 0


# --- MAIN ---
def main():
    print(f"--- CARGANDO MODELO {MODEL_NAME} ---")
    model = YOLO(MODEL_NAME)

    print(f"--- CONECTANDO CAMARA RTSP ---")
    cam = CameraStream(RTSP_URL).start()

    # Damos un segundo para que el hilo capture el primer frame
    time.sleep(1.5)

    ret, setup_frame = cam.read()
    if setup_frame is None:
        print("Error: No se pudo obtener imagen de la cámara. Verifica IP/Red.")
        cam.stop()
        return

    # --- FASE 1: DEFINICIÓN DE ZONAS (Sobre frame estático) ---
    print("Iniciando configuración de zonas...")

    # Zona 1: Bahía
    zona_bahia = get_zone_on_static_frame(setup_frame, "CONFIG: BAHIA CARGA",
                                          "Dibuja el area donde estacionan los camiones")
    if zona_bahia is None:
        cam.stop()
        return

    # Zona 2: Puerta
    zona_puerta = get_zone_on_static_frame(setup_frame, "CONFIG: PUERTA ALMACEN",
                                           "Dibuja el area de la PUERTA/ACCESO")
    if zona_puerta is None:
        cam.stop()
        return

    # --- FASE 2: BUCLE DE VIGILANCIA ---
    print("--- INICIANDO SISTEMA DE MONITOREO EN TIEMPO REAL ---")

    # Variables de estado (KPIs)
    tiempo_en_bahia = 0.0
    vehiculo_detectado = False
    tipo_vehiculo_actual = "---"

    # Clases YOLO (COCO dataset): 0:Persona, 2:Auto, 5:Bus, 7:Camión
    target_classes = [0, 2, 5, 7]

    prev_time = time.time()

    while True:
        # 1. Obtener frame más reciente (Bufferless)
        ret, frame = cam.read()
        if not ret: break

        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        # Preparamos copia para dibujar superposiciones
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # 2. Inferencia YOLO11s
        results = model(frame, classes=target_classes, verbose=False, imgsz=640, conf=0.5)

        # 3. Análisis de Detecciones
        hay_vehiculo_bahia = False
        hay_persona_activa = False
        hay_persona_puerta = False
        nombre_vehiculo_temp = ""

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                # Punto de referencia: Centro inferior (pies para personas, llantas para autos)
                cx, cy = int((x1 + x2) / 2), int(y2)

                label_draw = ""
                color_draw = (255, 255, 255)

                # -- LÓGICA VEHÍCULOS --
                if cls in [2, 5, 7]:
                    if cls == 2:
                        txt_v = "Auto/Van"
                    elif cls == 7:
                        txt_v = "Camion"
                    else:
                        txt_v = "Transporte"

                    if is_inside((cx, cy), zona_bahia):
                        hay_vehiculo_bahia = True
                        nombre_vehiculo_temp = txt_v
                        color_draw = COLOR_BAHIA_OCUPADA
                        label_draw = f"{txt_v} (Cargando)"
                    else:
                        color_draw = (200, 200, 200)  # Vehículo fuera de zona
                        label_draw = txt_v

                # -- LÓGICA PERSONAS --
                elif cls == 0:
                    label_draw = "Operador"
                    color_draw = (255, 255, 0)

                    if is_inside((cx, cy), zona_bahia):
                        hay_persona_activa = True

                    if is_inside((cx, cy), zona_puerta):
                        hay_persona_puerta = True
                        label_draw = "Acceso"
                        color_draw = (0, 0, 255)

                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_draw, 2)
                if label_draw:
                    cv2.putText(frame, label_draw, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_draw, 1)

        # 4. Cálculo de KPIs
        if hay_vehiculo_bahia:
            tiempo_en_bahia += delta_time
            tipo_vehiculo_actual = nombre_vehiculo_temp
        else:
            # Reseteamos si se va el vehículo (o puedes poner un delay para evitar parpadeos)
            tiempo_en_bahia = 0.0
            tipo_vehiculo_actual = "---"

        # Regla de Seguridad: Persona en puerta sin vehículo bloqueando bahía
        alerta_seguridad = False
        if hay_persona_puerta and not hay_vehiculo_bahia:
            alerta_seguridad = True

        # 5. Visualización de Zonas (Transparencias)
        # Zona Bahía
        c_bahia = COLOR_BAHIA_OCUPADA if hay_vehiculo_bahia else COLOR_BAHIA_LIBRE
        cv2.polylines(overlay, [zona_bahia], True, c_bahia, 2)
        cv2.fillPoly(overlay, [zona_bahia], c_bahia)

        # Zona Puerta
        c_puerta = COLOR_PUERTA_RISK if alerta_seguridad else COLOR_PUERTA_SAFE
        cv2.polylines(overlay, [zona_puerta], True, c_puerta, 2)
        if alerta_seguridad:
            cv2.fillPoly(overlay, [zona_puerta], c_puerta)

        # Aplicar transparencia (30%)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # 6. DASHBOARD (Panel Lateral)
        # Fondo del panel
        panel_w = 350
        cv2.rectangle(frame, (width - panel_w, 0), (width, 230), COLOR_KPI_BG, -1)

        # Textos
        cv2.putText(frame, "CONTROL LOGISTICO", (width - panel_w + 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # KPI Estado
        est_txt = "OCUPADO" if hay_vehiculo_bahia else "DISPONIBLE"
        est_col = (0, 255, 255) if hay_vehiculo_bahia else (0, 255, 0)
        cv2.putText(frame, f"Estado: {est_txt}", (width - panel_w + 20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, est_col, 1)

        # KPI Vehículo
        cv2.putText(frame, f"Unidad: {tipo_vehiculo_actual}", (width - panel_w + 20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # KPI Tiempo
        mins = int(tiempo_en_bahia // 60)
        secs = int(tiempo_en_bahia % 60)
        cv2.putText(frame, f"Tiempo: {mins:02}:{secs:02} min", (width - panel_w + 20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Alerta
        if alerta_seguridad:
            cv2.rectangle(frame, (width - panel_w, 180), (width, 230), (0, 0, 255), -1)
            cv2.putText(frame, "¡PUERTA VULNERABLE!", (width - panel_w + 20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            msg = "Operacion Segura" if hay_vehiculo_bahia else "Esperando arribo"
            col = (0, 255, 0) if hay_vehiculo_bahia else (200, 200, 200)
            cv2.putText(frame, msg, (width - panel_w + 20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)

        # FPS
        fps = 1.0 / delta_time if delta_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Monitor Almacen - YOLO11 RTSP", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("Sistema detenido.")


if __name__ == "__main__":
    main()