from http.server import BaseHTTPRequestHandler

from _common import image_features, read_body, send_json


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = read_body(self)
        filename = self.headers.get("X-Filename", "image.png")
        try:
            send_json(self, 200, image_features(raw, filename))
        except Exception as exc:
            send_json(self, 400, {"error": str(exc)})
