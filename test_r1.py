import asyncio
from fastapi.testclient import TestClient
from web_server import app

client = TestClient(app)
r1 = client.get("/api/v1/food_chain_timeline?lat=25&lon=130&species=yellowfin", headers={"X-API-Key": "dev-key-change-me"})
r2 = client.get("/api/v1/feeding_windows?lat=25&lon=130&species=yellowfin", headers={"X-API-Key": "dev-key-change-me"})
r3 = client.get("/api/v1/fish_movement?lat=25&lon=130&species=yellowfin", headers={"X-API-Key": "dev-key-change-me"})

print(f"timeline: {r1.status_code}")
print(f"windows: {r2.status_code}")
print(f"movement: {r3.status_code}")
