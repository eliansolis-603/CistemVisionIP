import cv2
import numpy as np
import mss
from ultralytics import YOLO
import time

# --- CONFIGURACIÓN ---
# Ajusta esto SOLO si la detección automática de ventana falla
# Usa Cmd+Shift+4 para medir la zona donde se ve el video en tu pantalla
REGION_MANUAL = {"top": 100, "left": 100, "width": 800, "height": 600}


def main():
    print("🚀 Cargando cerebro (YOLOv8)...")
    model = YOLO('yolov8n.pt')

    sct = mss.mss()
    print("✅ Sistema listo. Pon la App de LifeSmart en modo 'Cuadrícula 4 cámaras'.")
    print("ℹ️  Presiona 'q' para salir.")

    while True:
        # 1. Capturar Video (De la pantalla)
        # Si usas scrcpy, pon la ventana en una posición fija y usa REGION_MANUAL
        screenshot = np.array(sct.grab(REGION_MANUAL))

        # Limpiar imagen (Quitar canal Alpha)
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # 2. DIVIDIR LA IMAGEN (Simulando 4 IPs diferentes)
        # Asumimos que la pantalla está dividida en 2x2
        height, width, _ = frame.shape
        mid_x, mid_y = width // 2, height // 2

        # Cortar los 4 pedazos
        camaras = {
            "Cam 1 (Sup-Izq)": frame[0:mid_y, 0:mid_x],
            "Cam 2 (Sup-Der)": frame[0:mid_y, mid_x:width],
            "Cam 3 (Inf-Izq)": frame[mid_y:height, 0:mid_x],
            "Cam 4 (Inf-Der)": frame[mid_y:height, mid_x:width]
        }

        # 3. Analizar cada cámara
        for nombre, recorte_camara in camaras.items():
            # Si el recorte está vacío o negro, saltar
            if recorte_camara.size == 0: continue

            # Detectar Personas (Clase 0)
            results = model(recorte_camara, classes=[0], conf=0.5, verbose=False)

            # Dibujar detecciones sobre el recorte
            annotated_frame = results[0].plot()

            # Mostrar ventana individual
            cv2.imshow(nombre, annotated_frame)

        # Salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()