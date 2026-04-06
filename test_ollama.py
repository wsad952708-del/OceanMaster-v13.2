import urllib.request
import json
import time

url = "http://127.0.0.1:11434/api/chat"
data = json.dumps({
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "say hello"}],
    "stream": False
}).encode("utf-8")

req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

print("Warming up llama3.2...")
start = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start
        content = result.get("message", {}).get("content", "NO CONTENT")
        print(f"OK in {elapsed:.1f}s: {content[:100]}")
except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.1f}s: {e}")
