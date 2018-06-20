import cgi
import json
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

from app import get_image_from_base64, get_image_classificators


class HttpProcessor(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        # self.send_header('Content-type', 'text/html')

        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        self.send_header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')

        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers['Content-Type']
            })

        base64_str = form.getvalue("image")
        img = get_image_from_base64(base64_str)
        res = get_image_classificators(img, 100)

        dumps = json.dumps(res)

        self._set_headers()
        self.wfile.write(dumps)


if __name__ == '__main__':
    import sys

    port = int(sys.argv[1]) if len(sys.argv) >= 2 else 8080
    server = HTTPServer(("0.0.0.0", port), HttpProcessor)
    server.serve_forever()
