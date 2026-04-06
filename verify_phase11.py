"""Quick verification of Phase 11 fixes."""
import os, re

BASE = os.path.join(os.environ["USERPROFILE"], "Desktop", "好像快好了", "OceanMaster_v12_Release")

def read(f):
    return open(os.path.join(BASE, f), "r", encoding="utf-8").read()

# A1: No COPY .env in Dockerfile
c = read("Dockerfile.production")
a1 = not any(l.strip().startswith("COPY") and ".env" in l for l in c.splitlines())
print(f"A1 Dockerfile no .env COPY: {'PASS' if a1 else 'FAIL'}")

# A2: VERSION = 12.0
c = read("main_v10_3.py")
m = re.search(r'VERSION\s*=\s*"([^"]+)"', c)
print(f"A2 VERSION: {m.group(1) if m else 'NOT FOUND'} {'PASS' if m and m.group(1) == '12.0' else 'FAIL'}")

# A3: CHANGELOG has post-fix R²
c = read("CHANGELOG_v12.md")
a3 = "0.8826" in c and "0.9036" in c and "0.9019" in c and "0.9084" in c
print(f"A3 CHANGELOG metrics: {'PASS' if a3 else 'FAIL'}")

# A4: No bad rate_check calls
c = read("web_server.py")
calls = re.findall(r'_rate_check\(([^)]+)\)', c)
bad = [a for a in calls if "str(lat)" in a or "dvm_" in a]
print(f"A4 rate_check: {'PASS' if not bad else 'FAIL ' + str(bad)}")

# B1: test suite exists
b1 = os.path.exists(os.path.join(BASE, "tests", "test_v12_full.py"))
print(f"B1 test suite: {'PASS' if b1 else 'FAIL'}")

# B2: README has expanded limitations
c = read("README.md")
b2 = "SST feature importance" in c and "catch probability" in c
print(f"B2 README limitations: {'PASS' if b2 else 'FAIL'}")

# B3: WCPFC loader exists
b3 = os.path.exists(os.path.join(BASE, "engine", "ml", "wcpfc_data_loader.py"))
print(f"B3 WCPFC loader: {'PASS' if b3 else 'FAIL'}")

# Extra: DEPLOYMENT.md security warning
c = read("DEPLOYMENT.md")
ex1 = "Never bake credentials" in c
print(f"DEPLOYMENT.md security: {'PASS' if ex1 else 'FAIL'}")

# Extra: CHANGELOG Phase 11
c = read("CHANGELOG_v12.md")
ex2 = "Phase 11" in c
print(f"CHANGELOG Phase 11: {'PASS' if ex2 else 'FAIL'}")

all_pass = all([a1, m and m.group(1) == "12.0", a3, not bad, b1, b2, b3, ex1, ex2])
print(f"\n{'='*40}")
print(f"OVERALL: {'ALL PASS ✅' if all_pass else 'SOME FAILED ❌'}")
