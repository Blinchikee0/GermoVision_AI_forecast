from http.server import BaseHTTPRequestHandler

from _common import REFERENCES, WORLD_CITIES, send_json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        send_json(self, 200, {
            "ok": True,
            "references": [
                {"id": k, "name": v["name"], "disease": v["disease"], "length": len(v["aa"])}
                for k, v in REFERENCES.items()
            ],
            "cities": WORLD_CITIES,
        })
