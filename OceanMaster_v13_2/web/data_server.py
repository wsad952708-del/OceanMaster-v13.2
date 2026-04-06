"""
OceanMaster — Live Data Server
Serves hotspot JSON + static dashboard HTML
Dashboard polls /api/hotspots every 30s for live updates
"""
import http.server
import json
import os
import glob
import time
from urllib.parse import urlparse

PORT = 8081
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(WEB_DIR), 'output')


def get_latest_hotspots():
    """Find and return the most recent hotspots JSON."""
    patterns = [
        os.path.join(OUTPUT_DIR, 'hotspots.json'),
        os.path.join(OUTPUT_DIR, 'hotspots_*.json'),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return []
    latest = max(files, key=os.path.getmtime)
    with open(latest, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


class LiveHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        # API endpoint: latest hotspots
        if parsed.path == '/api/hotspots':
            try:
                data = get_latest_hotspots()
                payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                self.send_error(500, str(e))
            return

        # API endpoint: server status
        if parsed.path == '/api/status':
            status = {
                'server': 'OceanMaster Live Data Server',
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'hotspot_count': len(get_latest_hotspots()),
                'output_dir': OUTPUT_DIR,
            }
            payload = json.dumps(status, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # Otherwise serve static files
        super().do_GET()

    def log_message(self, format, *args):
        # Cleaner log
        if '/api/' in str(args[0]):
            print(f"[API] {args[0]}")


if __name__ == '__main__':
    with http.server.HTTPServer(('', PORT), LiveHandler) as httpd:
        print(f"OceanMaster Live Server on http://localhost:{PORT}")
        print(f"Dashboard: http://localhost:{PORT}/dashboard_v3.html")
        print(f"API: http://localhost:{PORT}/api/hotspots")
        print(f"Hotspots from: {OUTPUT_DIR}")
        httpd.serve_forever()
