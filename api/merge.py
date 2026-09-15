from http.server import BaseHTTPRequestHandler

from _common import merge_analysis, read_json, send_json


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = read_json(self)
            send_json(self, 200, merge_analysis(body.get("analysis", {}), body.get("image", {})))
        except Exception as exc:
            send_json(self, 400, {"error": str(exc)})
