import cv2
import time
from onvif import ONVIFCamera
import sys

# --- CONFIGURACIÓN ---
IP = '192.168.1.215'
ONVIF_PORT = 2020
RTSP_PORT = 554
USER = 'admin3'
PASS = 'admin123'

# CONFIGURACIÓN DE MOVIMIENTO FINO
MOVE_SPEED = 0.5  # Velocidad del motor (0.0 a 1.0). Bajamos a 0.5 para más precisión.
STEP_DURATION = 0.15  # Cuánto tiempo se mueve la cámara por cada pulsación (en segundos).

# 1. CONEXIÓN ONVIF
print(f"Conectando a controles ONVIF...")
try:
    mycam = ONVIFCamera(IP, ONVIF_PORT, USER, PASS)
    media = mycam.create_media_service()
    ptz = mycam.create_ptz_service()
    media_profile = media.GetProfiles()[0]

    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = media_profile.token
    stop_request = ptz.create_type('Stop')
    stop_request.ProfileToken = media_profile.token

    # Pre-configuramos el objeto de parada para que sea más rápido
    stop_request.PanTilt = True
    stop_request.Zoom = True

    print("✅ Controles listos.")
except Exception as e:
    print(f"❌ Error ONVIF: {e}")
    sys.exit()


# FUNCIÓN DE "PASO" (NUDGE)
def move_step(x, y):
    """
    Mueve la cámara brevemente y la detiene automáticamente.
    """
    try:
        # 1. Enviar comando de mover
        request.Velocity = {'PanTilt': {'x': x * MOVE_SPEED, 'y': y * MOVE_SPEED}}
        ptz.ContinuousMove(request)

        # 2. Esperar el tiempo del paso (esto pausa el video un instante muy breve)
        time.sleep(STEP_DURATION)

        # 3. Enviar comando de parar inmediatamente
        ptz.Stop(stop_request)
    except Exception as e:
        print(f"Error moviendo: {e}")


# 2. VIDEO
rtsp_url = f"rtsp://{USER}:{PASS}@{IP}:{RTSP_PORT}/stream1"
cap = cv2.VideoCapture(rtsp_url)

print("\n--- MODO PRECISIÓN ---")
print(" Presiona W/A/S/D para dar 'pasos' pequeños.")
print(" Presiona Q para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Pérdida de señal de video.")
        break

    frame = cv2.resize(frame, (960, 540))
    cv2.imshow('Tapo Control Fino', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        move_step(0, 1.0)  # Arriba
    elif key == ord('s'):
        move_step(0, -1.0)  # Abajo
    elif key == ord('a'):
        move_step(-1.0, 0)  # Izquierda
    elif key == ord('d'):
        move_step(1.0, 0)  # Derecha
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()