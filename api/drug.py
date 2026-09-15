from http.server import BaseHTTPRequestHandler

from _common import rank_drugs, read_json, send_json


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = read_json(self)
            if "analysis" not in body:
                send_json(self, 400, {"error": "post the analysis object under 'analysis'"}); return
            send_json(self, 200, rank_drugs(body["analysis"]))
        except Exception as exc:
            send_json(self, 400, {"error": str(exc)})
