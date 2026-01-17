import socket
import cv2
import time

# Configuración basada en tu video anterior
TARGET_IP = "192.168.1.62"
TIMEOUT = 4  # Segundos de espera

# Puertos comunes para cámaras IP
# 554: RTSP estándar (El más probable)
# 8554: RTSP alternativo
# 80: Web/HTTP
# 8000: SDK/Control
# 8080: ONVIF/Web alternativo
PORTS_TO_CHECK = [554, 8554, 80, 8000, 8080, 1935]


def check_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT)
    result = sock.connect_ex((ip, port))
    sock.close()
    return result == 0


def test_rtsp_connection(url):
    print(f"   Probando flujo: {url} ...", end="", flush=True)
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(" ✅ ¡ÉXITO! Video recibido.")
            return True
    print(" ❌ Falló.")
    return False


print(f"--- Iniciando diagnóstico para LifeSmart Cam ({TARGET_IP}) ---")

open_ports = []
print("1. Escaneando puertos abiertos...")
for port in PORTS_TO_CHECK:
    if check_port(TARGET_IP, port):
        print(f"   [ABIERTO] Puerto {port} detectado.")
        open_ports.append(port)
    else:
        print(f"   [CERRADO] Puerto {port}")

if not open_ports:
    print("\n⚠️ ALERTA: No se encontraron puertos abiertos.")
    print("   Posible causa: La cámara entró en modo 'sleep' o la IP cambió.")
    print("   Solución: Abre la App de LifeSmart para 'despertar' la cámara y reintenta.")
else:
    print(f"\n2. Puertos encontrados: {open_ports}. Probando URLs comunes...")

    # Aquí debes poner la contraseña que configuraste en "Local Password Settings"
    # Si no la has puesto, ve a la App -> Ajustes -> Local Password y pon una (ej. admin123)
    PASSWORD_LOCAL = "admin123"  # <--- CAMBIA ESTO
    USERNAME = "admin"

    # Lista de URLs probables basada en hardware LifeSmart/Tuya
    urls_to_try = []

    if 554 in open_ports or 8554 in open_ports:
        port = 554 if 554 in open_ports else 8554
        urls_to_try.append(f"rtsp://{USERNAME}:{PASSWORD_LOCAL}@{TARGET_IP}:{port}/live/ch0")
        urls_to_try.append(f"rtsp://{USERNAME}:{PASSWORD_LOCAL}@{TARGET_IP}:{port}/onvif1")
        urls_to_try.append(f"rtsp://{USERNAME}:{PASSWORD_LOCAL}@{TARGET_IP}:{port}/h264_stream")

    found = False
    for url in urls_to_try:
        if test_rtsp_connection(url):
            print(f"\n🎉 URL CONFIRMADA: {url}")
            print("Copia esta URL y úsala en tu script principal.")
            found = True
            break

    if not found:
        print("\n❌ Los puertos están abiertos pero las URLs estándar fallaron.")
        print("   Asegúrate de que la 'Local Password' sea correcta.")
        print("   Nota: NO es la misma contraseña de tu cuenta LifeSmart, es la del menú 'Local Password'.")