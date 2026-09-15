from http.server import BaseHTTPRequestHandler

from _common import analyze, read_body, send_json


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = read_body(self)
        filename = self.headers.get("X-Filename", "input.fasta")
        ref_id = self.headers.get("X-Reference", "sars2_spike")
        try:
            send_json(self, 200, analyze(raw, filename, ref_id))
        except Exception as exc:
            send_json(self, 400, {"error": str(exc)})
