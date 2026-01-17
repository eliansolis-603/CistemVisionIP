import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests  # asegúrate de instalarlo con: pip install requests


class LifeSmartProxyHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        # Permitir CORS desde cualquier origen (para pruebas)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        # Preflight CORS
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path != "/proxy":
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Leer body enviado por el navegador
        length = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "ok": False,
                "error": "Invalid JSON in proxy request",
                "detail": str(e),
            }
            self.wfile.write(json.dumps(resp, indent=2).encode("utf-8"))
            return

        # Esperamos algo del estilo:
        # {
        #   "url": "https://api.us.ilifesmart.com/app/api.EpGetAllAgts",
        #   "body": { ... }   # o string JSON
        # }
        target_url = payload.get("url")
        target_body = payload.get("body")

        if not target_url:
            self.send_response(400)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {"ok": False, "error": "Missing 'url' field in proxy payload"}
            self.wfile.write(json.dumps(resp, indent=2).encode("utf-8"))
            return

        # Si el body viene como string, lo usamos tal cual.
        # Si viene como objeto, lo serializamos.
        if isinstance(target_body, str):
            data_to_send = target_body
        else:
            data_to_send = json.dumps(target_body or {})

        print(f"[PROXY] Forwarding POST to {target_url}")
        # Hacer la petición real a LifeSmart
        try:
            ls_resp = requests.post(
                target_url,
                data=data_to_send,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )

            # Devolver al navegador el mismo status y texto
            self.send_response(ls_resp.status_code)
            self._set_cors_headers()
            self.send_header("Content-Type", ls_resp.headers.get("Content-Type", "text/plain"))
            self.end_headers()
            self.wfile.write(ls_resp.content)

        except requests.RequestException as e:
            # Error de red al hablar con LifeSmart
            self.send_response(502)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "ok": False,
                "error": "Error calling LifeSmart API",
                "detail": str(e),
            }
            self.wfile.write(json.dumps(resp, indent=2).encode("utf-8"))


def run(server_address=("0.0.0.0", 7001)):
    httpd = HTTPServer(server_address, LifeSmartProxyHandler)
    print(f"LifeSmart proxy listening on http://{server_address[0]}:{server_address[1]}/proxy")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping proxy...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()
