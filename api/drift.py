from http.server import BaseHTTPRequestHandler

from _common import drift, read_json, send_json


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = read_json(self)
            res = drift(
                int(body.get("seed_city", 0)),
                float(body.get("r0", 2.5)),
                float(body.get("generation_time", 5.0)),
                int(body.get("days", 90)),
                float(body.get("mutation_rate", 0.03)),
                body.get("reference", "sars2_spike"),
                body.get("analysis"),
            )
            send_json(self, 200, res)
        except Exception as exc:
            send_json(self, 400, {"error": str(exc)})
