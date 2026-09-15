from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from _common import REFERENCES, send_json, synth_window


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        ref_id = (qs.get("ref", ["sars2_spike"]) or ["sars2_spike"])[0]
        try:
            rate = float((qs.get("rate", ["0.04"]) or ["0.04"])[0])
        except ValueError:
            rate = 0.04
        try:
            body = synth_window(ref_id, rate).encode("utf-8")
        except KeyError:
            send_json(self, 404, {"error": "unknown reference"}); return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
