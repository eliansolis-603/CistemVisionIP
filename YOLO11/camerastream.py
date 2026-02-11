import cv2
import numpy as np
from vidgear.gears import CamGear
import time

# URL RTSP
RTSP_URL = "rtsp://admin123:admin123@192.168.1.228:554/stream1"

def main():
    print(f"--- INICIANDO CON CAMGEAR (Motor FFmpeg) ---")
    print(f"Conectando a: {RTSP_URL}")

    # CamGear opciones:
    # - STREAM_RESOLUTION: Baja la resolución para ganar velocidad
    # - STREAM_PARAMS: Forzamos TCP para evitar errores en WiFi
    options = {
        "CAP_PROP_FRAME_WIDTH": 640,
        "CAP_PROP_FRAME_HEIGHT": 480,
        "RTSP_TRANSPORT": "tcp"
    }

    try:
        # CamGear maneja la conexión de forma mucho más robusta que cv2 nativo
        stream = CamGear(source=RTSP_URL, logging=True, **options).start()
    except Exception as e:
        print(f"Error fatal conectando: {e}")
        return

    print(">> CONEXIÓN EXITOSA. Presiona 'q' para salir.")

    while True:
        # Leer frame
        frame = stream.read()

        # Si frame es None, algo pasó
        if frame is None:
            break

        # Aquí iría tu lógica de YOLO (dibujamos un círculo para probar)
        cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
        cv2.putText(frame, "CamGear RTSP", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Prueba VidGear", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()