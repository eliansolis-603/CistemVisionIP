import requests
import hashlib
import time
import json

# --- TUS CREDENCIALES (Ya integradas) ---
APP_KEY = "m4s5Ht6ZfuCnxHBtM7MlJQ"
APP_TOKEN = "EgZyBnkU4oOPzcycDHlouw"
USER_ID = "8450498"
USER_TOKEN = "LRTnkh3aZMhmwyRsB1u2Kg"

# 1. CORRECCIÓN: Usamos el servidor de EE.UU. (detectado en tu proxy)
BASE_URL = "https://api.us.ilifesmart.com/app"


def get_sign(method, time_now, params):
    """Genera la firma MD5 de seguridad"""
    base = f"method:{method}"
    if params:
        for k in sorted(params.keys()):
            base += f",{k}:{str(params[k])}"  # Aseguramos que sea string
    base += f",time:{time_now},userid:{USER_ID},usertoken:{USER_TOKEN},appkey:{APP_KEY},apptoken:{APP_TOKEN}"
    return hashlib.md5(base.encode()).hexdigest()


def call_api(method, params={}):
    """Función genérica para llamar a cualquier método de la API"""
    url = f"{BASE_URL}/api.{method}"
    time_now = int(time.time())
    sign = get_sign(method, time_now, params)

    payload = {
        "id": 1,
        "method": method,
        "params": params,
        "system": {
            "ver": "1.0",
            "lang": "en",
            "userid": USER_ID,
            "appkey": APP_KEY,
            "time": time_now,
            "sign": sign
        }
    }

    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        print(f"❌ Error de conexión en {method}: {e}")
        return None


def main():
    print(f"📡 Conectando a {BASE_URL}...")

    # PASO 1: Obtener la lista de TODOS los dispositivos
    # Probamos 'EpGetAll' (Dispositivos finales)
    resp = call_api("EpGetAll")

    cameras_found = []

    if resp and resp.get('code') == 0:
        print(f"✅ Conexión exitosa. Analizando {len(resp.get('result', []))} dispositivos...")

        for item in resp.get('result', []):
            # Imprimimos todo lo que parece cámara para que veas qué detecta
            dev_type = item.get('type', '')
            dev_name = item.get('name', 'Sin Nombre')
            dev_id = item.get('me')  # EL ID IMPORTANTE

            # Filtro: Buscamos tipos comunes de cámaras
            if 'cam' in dev_type.lower() or 'video' in dev_type.lower():
                print(f"🔎 Cámara detectada: {dev_name} (ID: {dev_id})")
                cameras_found.append(dev_id)
    else:
        print(f"⚠️ La lista general falló o está vacía: {resp}")

    # Si no encontró nada en EpGetAll, a veces las cámaras IP cuentan como 'Agentes'
    if not cameras_found:
        print("🤔 No se vieron cámaras en la lista estándar. Probando lista de Estaciones (Agts)...")
        resp_agt = call_api("EpGetAllAgts")
        if resp_agt and resp_agt.get('code') == 0:
            for item in resp_agt.get('result', []):
                # A veces la cámara IP es su propia estación
                print(f"   Posible Cámara/Estación: {item.get('name')} (ID: {item.get('agt')})")
                # Agregamos para probar suerte
                cameras_found.append(item.get('agt'))

    # PASO 2: Obtener el link de video específico para cada cámara encontrada
    print("\n🎥 Buscando enlaces de video (RTSP)...")
    final_links = []

    for cam_id in cameras_found:
        # Preguntamos DETALLES de este ID específico
        # Aquí es donde suele aparecer la URL oculta
        details = call_api("EpGet", {"me": cam_id})

        if details and details.get('code') == 0:
            data = details.get('result', {}).get('data', {})

            # Buscamos variaciones comunes del link
            rtsp = data.get('RTSP_URL') or data.get('url') or data.get('stream_addr') or data.get('L')

            if rtsp:
                print(f"✅ LINK OBTENIDO ({cam_id}): {rtsp}")
                final_links.append(rtsp)
            else:
                print(f"⚠️ Se obtuvo info de {cam_id}, pero no campo de video explícito.")
                print(f"   Datos recibidos: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Falló consulta de detalles para {cam_id}")

    print("\n--- RESULTADO FINAL: COPIA ESTO AL SCRIPT DE VISIÓN ---")
    print(final_links)


if __name__ == "__main__":
    main()