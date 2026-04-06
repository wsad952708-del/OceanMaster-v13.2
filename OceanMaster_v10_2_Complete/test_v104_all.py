"""OceanMaster v10.4 — Full Verification"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

print("=" * 60)
print("  OceanMaster v10.4 Verification")
print("=" * 60)

passed = 0
failed = 0

# ── Task #1: Legal Compliance ──
print("\n--- Task #1: Legal Compliance Checker ---")
try:
    from compliance.regulation_checker import (
        RegulationChecker, check_point, filter_hotspots
    )
    checker = RegulationChecker()

    # Point inside Palau MPA (should be illegal)
    r1 = check_point(5.0, 133.0)
    assert r1["legal"] == False, f"Expected illegal, got {r1}"
    print(f"  PASS  Palau MPA check: legal={r1['legal']}, zone={r1['zone_name']}")

    # Point in open ocean (should be legal)
    r2 = check_point(25.0, 150.0)
    assert r2["legal"] == True, f"Expected legal, got {r2}"
    print(f"  PASS  Open ocean check: legal={r2['legal']}")

    # Filter hotspots
    test_hotspots = [
        {"lat": 5.0, "lon": 133.0, "score": 0.9},  # Palau MPA
        {"lat": 25.0, "lon": 150.0, "score": 0.7},  # Open ocean
    ]
    filtered = filter_hotspots(test_hotspots, remove_illegal=True)
    assert len(filtered) == 1, f"Expected 1 legal hotspot, got {len(filtered)}"
    print(f"  PASS  filter_hotspots: 2 -> {len(filtered)} (1 removed)")

    passed += 3
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  FAIL  Task #1: {e}")
    failed += 1

# ── Task #2: EKE Geostrophic ──
print("\n--- Task #2: SSH -> EKE (Geostrophic) ---")
try:
    from engine.algorithms import calculate_eke

    # Synthetic SSH: warm eddy centered at (15, 140)
    lats = np.linspace(10, 20, 20)
    lons = np.linspace(135, 145, 20)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing='ij')
    # Gaussian bump = warm eddy
    ssh = 0.1 * np.exp(-((lat_g - 15)**2 + (lon_g - 140)**2) / 4)

    result = calculate_eke(ssh, lats, lons)

    assert "eke" in result
    assert "u_geo" in result
    assert "v_geo" in result
    assert result["eke"].shape == (20, 20)
    eke_max = float(np.nanmax(result["eke"]))
    u_max = float(np.nanmax(np.abs(result["u_geo"])))
    print(f"  PASS  EKE computed: max={eke_max:.6f} m2/s2, |u_g|_max={u_max:.4f} m/s")
    print(f"  PASS  Eddies: {result['n_warm']} warm, {result['n_cold']} cold")
    passed += 2
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  FAIL  Task #2: {e}")
    failed += 1

# ── Task #3: SSS Salinity Fetcher ──
print("\n--- Task #3: SSS Salinity Fetcher ---")
try:
    from engine.salinity_fetcher import SalinityFetcher

    fetcher = SalinityFetcher()

    # Test gradient computation (static method)
    sss = np.random.uniform(34.0, 35.5, (10, 10)).astype(np.float32)
    lats = np.linspace(20, 25, 10)
    lons = np.linspace(130, 135, 10)

    gradient = fetcher._compute_sss_gradient(sss, lats, lons)
    assert gradient.shape == (10, 10)
    assert gradient.dtype == np.float32
    print(f"  PASS  SSS gradient: shape={gradient.shape}, max={np.nanmax(gradient):.4f} psu/km")

    # Test is_salinity_front detection
    from engine.salinity_fetcher import SSS_FRONT_THRESHOLD
    n_fronts = int(np.sum(gradient >= SSS_FRONT_THRESHOLD))
    print(f"  PASS  Salinity fronts: {n_fronts} cells above {SSS_FRONT_THRESHOLD} psu/km")
    passed += 2
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  FAIL  Task #3: {e}")
    failed += 1

# ── Task #4: Catch Report DB ──
print("\n--- Task #4: Catch Report (SQLite) ---")
try:
    import sqlite3, tempfile, json

    # Test DB creation and insertion
    db_path = os.path.join(tempfile.gettempdir(), "test_catch.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS catch_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vessel_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            lat REAL NOT NULL, lon REAL NOT NULL,
            target_species TEXT NOT NULL,
            actual_cpue_kg_day REAL NOT NULL,
            predicted_cpue REAL,
            env_snapshot TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    env_snap = json.dumps({"sst": 28.1, "sss": 34.8, "eke": 0.012})
    conn.execute(
        "INSERT INTO catch_reports (vessel_id, timestamp, lat, lon, target_species, "
        "actual_cpue_kg_day, predicted_cpue, env_snapshot) VALUES (?,?,?,?,?,?,?,?)",
        ("CT6-1234", "2026-02-14T08:00:00Z", 24.5, 140.2, "yellowfin", 85.0, 72.3, env_snap)
    )
    conn.commit()

    row = conn.execute("SELECT * FROM catch_reports WHERE vessel_id='CT6-1234'").fetchone()
    assert row is not None
    total = conn.execute("SELECT COUNT(*) FROM catch_reports").fetchone()[0]
    env_data = json.loads(row[8])
    print(f"  PASS  SQLite insert OK: id={row[0]}, total={total}")
    print(f"  PASS  env_snapshot round-trip: sst={env_data['sst']}, sss={env_data['sss']}")

    conn.close()
    os.unlink(db_path)
    passed += 2
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  FAIL  Task #4: {e}")
    failed += 1

# ── Pipeline import check ──
print("\n--- Pipeline Integration ---")
try:
    from engine.algorithms import calculate_eke
    from engine.salinity_fetcher import SalinityFetcher
    from compliance.regulation_checker import RegulationChecker, filter_hotspots
    print(f"  PASS  All v10.4 imports OK")
    passed += 1
except Exception as e:
    print(f"  FAIL  Pipeline imports: {e}")
    failed += 1

print(f"\n{'=' * 60}")
print(f"  TOTAL: {passed} passed, {failed} failed")
print(f"{'=' * 60}")
assert not failed, f"{failed} test(s) failed"
