import requests
import hashlib
import time
import json

# --- TUS CREDENCIALES (Recuperadas de tu mensaje anterior) ---
APP_KEY = "m4s5Ht6ZfuCnxHBtM7MlJQ"
APP_TOKEN = "EgZyBnkU4oOPzcycDHlouw"
USER_ID = "8450498"
USER_TOKEN = "LRTnkh3aZMhmwyRsB1u2Kg"

# ¡AQUÍ ESTABA EL SECRETO! Usamos el servidor descubierto en tus logs
# Probamos primero el apimux (Multiplexor) y luego el global por si acaso
POSSIBLE_SERVERS = [
    "https://apimux.ilifesmart.com/app",  # El que salió en tu log
    "https://api.ilifesmart.com/app",  # Global
    "https://api.us.ilifesmart.com/app"  # USA Backup
]


def get_sign(method, time_now, params):
    """Genera la firma MD5 obligatoria"""
    base = f"method:{method}"
    if params:
        for k in sorted(params.keys()):
            base += f",{k}:{str(params[k])}"
    base += f",time:{time_now},userid:{USER_ID},usertoken:{USER_TOKEN},appkey:{APP_KEY},apptoken:{APP_TOKEN}"
    return hashlib.md5(base.encode()).hexdigest()


def call_api(base_url, method, params={}):
    url = f"{base_url}/api.{method}"
    time_now = int(time.time())
    sign = get_sign(method, time_now, params)

    payload = {
        "id": 1, "method": method, "params": params,
        "system": {
            "ver": "1.0", "lang": "en", "userid": USER_ID,
            "appkey": APP_KEY, "time": time_now, "sign": sign
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        return response.json()
    except:
        return None


def main():
    print(f"🕵️ Iniciando búsqueda de cámaras para usuario {USER_ID}...")

    found_cameras = []
    working_server = None

    # 1. Encontrar cuál servidor responde
    for server in POSSIBLE_SERVERS:
        print(f"📡 Probando servidor: {server} ... ", end="")
        resp = call_api(server, "EpGetAll")

        if resp and resp.get('code') == 0:
            print("✅ ¡CONECTADO!")
            working_server = server
            devices = resp.get('result', [])
            print(f"   └── Se encontraron {len(devices)} dispositivos.")

            # Buscar cámaras en este servidor
            for dev in devices:
                # Guardamos TODO lo que parezca cámara o estación
                if 'cam' in dev.get('type', '').lower() or 'smart' in dev.get('type', '').lower():
                    found_cameras.append(dev)
            break  # Ya encontramos el servidor bueno, dejamos de buscar
        else:
            error_msg = resp.get('msg') if resp else "Sin respuesta"
            print(f"❌ Falló ({error_msg})")

    if not working_server:
        print("\n⚠️ CRÍTICO: Ningún servidor aceptó las credenciales.")
        print("   Posible causa: El UserToken caducó. Genera uno nuevo con el HTML.")
        return

    # 2. Extraer URLs de video
    print("\n🎥 Analizando cámaras encontradas...")
    if not found_cameras:
        print("⚠️ No se vieron dispositivos tipo 'Cámara' en la lista principal.")
        print("   Intentando truco de 'Estaciones' (Agts)...")
        # A veces las cámaras están escondidas en EpGetAllAgts
        resp_agt = call_api(working_server, "EpGetAllAgts")
        if resp_agt:
            for agt in resp_agt.get('result', []):
                found_cameras.append({"me": agt['agt'], "name": "Posible Estación/Cámara"})

    valid_links = []
    for cam in found_cameras:
        cam_id = cam.get('me')
        print(f"   > Consultando detalles de ID: {cam_id} ({cam.get('name')})...")

        # Llamada específica para obtener atributos de video
        details = call_api(working_server, "EpGet", {"me": cam_id})

        if details and 'result' in details:
            data = details['result'].get('data', {})

            # Lista de atributos donde LifeSmart suele esconder el video
            # Basado en tu PDF 'Device Attribute List'
            candidates = [
                data.get('RTSP_URL'),
                data.get('stream_addr'),
                data.get('url'),
                data.get('L'),  # Stream local
                data.get('R')  # Stream remoto
            ]

            # Filtramos los que no sean None
            urls = [u for u in candidates if u and str(u).startswith(('rtsp', 'http', 'rtmp'))]

            if urls:
                print(f"   🎉 ¡EUREKA! URL encontrada: {urls[0]}")
                valid_links.append(urls[0])
            else:
                print(f"   ⚠️ Info obtenida pero sin video. Datos: {json.dumps(data)}")

    print("\n" + "=" * 40)
    print("RESUMEN DE ENLACES PARA TU MAC")
    print("=" * 40)
    if valid_links:
        for link in valid_links:
            print(f"VIDEO_URL = '{link}'")
        print("\nCopialos y pégalos en tu script de detección de personas.")
    else:
        print("No se extrajeron enlaces directos. Es posible que el video esté encriptado P2P.")


if __name__ == "__main__":
    main()