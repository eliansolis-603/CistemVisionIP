from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from urllib.parse import urlparse

# CONFIGURACIÓN
PORT = 7001


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.handle_request('POST')

    def do_GET(self):
        self.handle_request('GET')

    def handle_request(self, method):
        # 1. Leer el cuerpo de la petición original (lo que envía tu App)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        # 2. Determinar destino (Asumimos US por tus logs anteriores)
        # Si la App intenta conectar a otro lado, lo veremos aquí.
        target_url = "https://api.us.ilifesmart.com/app/api.EpGetAll"  # Default

        # A veces la App manda la URL destino en los headers o path
        # Para simplificar, este proxy reenvía todo a la API de US que detectamos antes
        real_target = "https://api.us.ilifesmart.com" + self.path

        print(f"\n📨 [SOLICITUD] App -> Nube: {self.path}")

        # 3. Reenviar la petición a la Nube Real
        headers = {k: v for k, v in self.headers.items() if k.lower() != 'host'}
        try:
            if method == 'POST':
                resp = requests.post(real_target, data=post_data, headers=headers)
            else:
                resp = requests.get(real_target, headers=headers)

            # 4. IMPRIMIR LA RESPUESTA (AQUÍ ESTÁ EL ORO)
            print(f"📦 [RESPUESTA] Nube -> App ({resp.status_code}):")
            try:
                data = resp.json()
                # Buscamos patrones de video automáticamente
                import json
                print(json.dumps(data, indent=2))

                # Alerta visual si encontramos lo que buscas
                text_dump = json.dumps(data)
                if "rtsp" in text_dump.lower() or "stream" in text_dump.lower():
                    print("\n🚨🚨 ¡ALERTA! URL DE VIDEO ENCONTRADA ARRIBA 🚨🚨\n")
            except:
                print(resp.text[:500])  # Si no es JSON, imprime texto plano

            # 5. Responder a la App para que no se queje
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.content)

        except Exception as e:
            print(f"❌ Error en el puente: {e}")
            self.send_error(500)


print(f"🕵️ PROXY ESPÍA LISTO en el puerto {PORT}")
print("1. Averigua la IP local de tu Mac (ej. 192.168.1.XX)")
print("2. Configura el WiFi de tu celular: Proxy Manual -> IP de Mac, Puerto 7001")
print("3. Abre la App LifeSmart y refresca la lista de cámaras.")
httpd = HTTPServer(('0.0.0.0', PORT), ProxyHandler)
httpd.serve_forever()