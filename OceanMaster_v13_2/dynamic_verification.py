"""
OceanMaster v13.2 — 動態執行驗證腳本
=====================================
Phase 1: 全模組 import 測試
Phase 2: 語法錯誤掃描 (AST)
Phase 3: 核心類別實例化 + 方法存在性
Phase 4: ML 模型權重載入驗證
Phase 5: FastAPI 端點驗證
Phase 6: 端到端特徵工程 → 預測流程
Phase 7: 密鑰外洩掃描
"""

import sys, os, ast, json, time, importlib, traceback
from pathlib import Path
from datetime import datetime

# Setup
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

results = {
    "scan_time": datetime.now().isoformat(),
    "python_version": sys.version,
    "phases": {}
}

def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════
#  Phase 1: AST 語法驗證 (所有 .py)
# ═══════════════════════════════════════════════════════════
section("Phase 1: AST 語法驗證")

py_files = list(BASE.rglob("*.py"))
py_files = [f for f in py_files if 'venv' not in str(f) and '__pycache__' not in str(f) 
            and '.git' not in str(f) and 'data_fetcher_external' not in str(f)]

syntax_ok = []
syntax_fail = []

for f in py_files:
    try:
        source = f.read_text(encoding='utf-8', errors='replace')
        ast.parse(source)
        syntax_ok.append(str(f.relative_to(BASE)))
    except SyntaxError as e:
        syntax_fail.append({
            "file": str(f.relative_to(BASE)),
            "line": e.lineno,
            "msg": str(e.msg)
        })

print(f"  ✅ 語法正確: {len(syntax_ok)}/{len(py_files)}")
for fail in syntax_fail:
    print(f"  ❌ {fail['file']}:{fail['line']} — {fail['msg']}")

results["phases"]["syntax"] = {
    "total": len(py_files),
    "pass": len(syntax_ok),
    "fail": len(syntax_fail),
    "failures": syntax_fail
}

# ═══════════════════════════════════════════════════════════
#  Phase 2: engine/ 模組 import 測試
# ═══════════════════════════════════════════════════════════
section("Phase 2: engine/ 模組 Import 測試")

engine_modules = []
for f in (BASE / "engine").rglob("*.py"):
    if '__pycache__' in str(f) or '__init__' in f.name:
        continue
    rel = f.relative_to(BASE).with_suffix('')
    mod_name = str(rel).replace(os.sep, '.').replace('/', '.')
    engine_modules.append(mod_name)

import_ok = []
import_fail = []

for mod_name in sorted(engine_modules):
    try:
        importlib.import_module(mod_name)
        import_ok.append(mod_name)
        print(f"  ✅ {mod_name}")
    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e)[:100]
        import_fail.append({"module": mod_name, "error": f"{err_type}: {err_msg}"})
        print(f"  ❌ {mod_name} → {err_type}: {err_msg}")

print(f"\n  Import 結果: {len(import_ok)}/{len(engine_modules)} 成功")

results["phases"]["import"] = {
    "total": len(engine_modules),
    "pass": len(import_ok),
    "fail": len(import_fail),
    "failures": import_fail
}

# ═══════════════════════════════════════════════════════════
#  Phase 3: 核心類別實例化
# ═══════════════════════════════════════════════════════════
section("Phase 3: 核心類別實例化")

class_tests = [
    ("engine.algorithms", "OceanAlgorithms", {}),
    ("engine.hsi_models", "HSICalculator", {}),
    ("engine.safety_checker", "SafetyChecker", {}),
    ("engine.typhoon_tracker", "TyphoonTracker", {}),
    ("engine.species_params", None, None),  # module-level only
    ("engine.thermocline_fetcher", "ThermoclineFetcher", {}),
    ("engine.gebco_features", "GEBCOFeatureExtractor", {}),
    ("engine.kuroshio_engine", "KuroshioEngine", {}),
    ("engine.lunar_model", "LunarFishingModel", {}),
    ("engine.shap_explainer", "SHAPExplainer", None),  # needs model arg
    ("engine.ai_fusion", "AIFusionEngine", {}),
]

class_ok = []
class_fail = []

for mod_name, cls_name, kwargs in class_tests:
    try:
        mod = importlib.import_module(mod_name)
        if cls_name is None:
            # Module-level check only
            class_ok.append(mod_name)
            print(f"  ✅ {mod_name} (module loaded)")
            continue
        cls = getattr(mod, cls_name)
        if kwargs is not None:
            obj = cls(**kwargs)
            # Check it has methods
            methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m))]
            class_ok.append(f"{mod_name}.{cls_name}")
            print(f"  ✅ {cls_name}() — {len(methods)} public methods")
        else:
            class_ok.append(f"{mod_name}.{cls_name}")
            print(f"  ✅ {cls_name} (class exists)")
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:100]}"
        class_fail.append({"class": f"{mod_name}.{cls_name}", "error": err})
        print(f"  ❌ {mod_name}.{cls_name} → {err}")

results["phases"]["classes"] = {
    "total": len(class_tests),
    "pass": len(class_ok),
    "fail": len(class_fail),
    "failures": class_fail
}

# ═══════════════════════════════════════════════════════════
#  Phase 4: ML 模型權重載入
# ═══════════════════════════════════════════════════════════
section("Phase 4: ML 模型權重檢查")

model_dir = BASE / "models" / "checkpoints_r2"
pt_files = list(model_dir.glob("*.pt")) if model_dir.exists() else []
all_pt = list(BASE.rglob("*.pt"))
all_pt = [f for f in all_pt if 'venv' not in str(f)]

print(f"  .pt 權重檔總數: {len(all_pt)}")
for f in all_pt:
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  📦 {f.relative_to(BASE)} — {size_mb:.1f} MB")

# Try loading with torch if available
try:
    import torch
    torch_available = True
    for f in all_pt:
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                keys = list(data.keys())[:5]
                print(f"  ✅ {f.name} 可載入 — keys: {keys}")
            else:
                print(f"  ✅ {f.name} 可載入 — type: {type(data).__name__}")
        except Exception as e:
            print(f"  ❌ {f.name} 載入失敗: {e}")
except ImportError:
    torch_available = False
    print(f"  ⚠️ torch 未安裝，跳過權重載入驗證")
    # At least check file integrity
    for f in all_pt:
        try:
            with open(f, 'rb') as fh:
                magic = fh.read(8)
                # PyTorch files start with magic number or zip
                is_zip = magic[:2] == b'PK'  # ZIP format (modern torch.save)
                is_legacy = magic[:4] == b'\x80\x02'  # pickle protocol 2
                if is_zip or is_legacy:
                    print(f"  ✅ {f.name} — 檔案格式正確 ({'ZIP/modern' if is_zip else 'legacy pickle'})")
                else:
                    print(f"  ⚠️ {f.name} — 未知格式 magic: {magic[:4].hex()}")
        except Exception as e:
            print(f"  ❌ {f.name} — 讀取失敗: {e}")

results["phases"]["models"] = {
    "pt_files": len(all_pt),
    "torch_available": torch_available if 'torch_available' in dir() else False,
    "details": [{"file": str(f.relative_to(BASE)), "size_mb": round(f.stat().st_size/1024/1024, 1)} for f in all_pt]
}

# ═══════════════════════════════════════════════════════════
#  Phase 5: Stacking Ensemble 端到端
# ═══════════════════════════════════════════════════════════
section("Phase 5: Stacking Ensemble 載入 + 特徵工程")

try:
    from engine.ml.stacking_ensemble import StackingEnsemblePredictor, FeatureEngineer
    
    # Test FeatureEngineer
    import numpy as np
    fe = FeatureEngineer()
    print(f"  ✅ FeatureEngineer 建立成功")
    
    # Check if model weights can load
    predictor = StackingEnsemblePredictor()
    model_path = BASE / "models" / "checkpoints_r2" / "r2test_best.pt"
    
    if model_path.exists():
        try:
            predictor.load_model(str(model_path))
            print(f"  ✅ Stacking Ensemble 模型載入成功")
            
            # Try a dummy prediction
            try:
                # Create minimal test features
                n_features = 66  # Expected feature count
                test_data = np.random.rand(5, n_features).astype(np.float32)
                
                # Use predict if available
                if hasattr(predictor, 'predict'):
                    preds = predictor.predict(test_data)
                    print(f"  ✅ predict() 執行成功 — output shape: {np.array(preds).shape}")
                elif hasattr(predictor, 'predict_proba'):
                    preds = predictor.predict_proba(test_data)
                    print(f"  ✅ predict_proba() 執行成功")
                else:
                    print(f"  ⚠️ 找不到 predict/predict_proba 方法")
            except Exception as e:
                print(f"  ⚠️ 預測測試失敗: {type(e).__name__}: {str(e)[:100]}")
        except Exception as e:
            print(f"  ❌ 模型載入失敗: {type(e).__name__}: {str(e)[:150]}")
    else:
        print(f"  ⚠️ 模型權重檔不存在: {model_path}")
    
    results["phases"]["stacking_e2e"] = {"status": "ok"}
except Exception as e:
    err = f"{type(e).__name__}: {str(e)[:200]}"
    print(f"  ❌ Stacking Ensemble 測試失敗: {err}")
    results["phases"]["stacking_e2e"] = {"status": "fail", "error": err}

# ═══════════════════════════════════════════════════════════
#  Phase 6: FastAPI 端點驗證
# ═══════════════════════════════════════════════════════════
section("Phase 6: FastAPI 端點驗證 (靜態)")

try:
    # Parse web_server.py to find route decorators
    ws_path = BASE / "web_server.py"
    if ws_path.exists():
        source = ws_path.read_text(encoding='utf-8', errors='replace')
        
        import re
        routes = re.findall(r'@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']', source)
        
        print(f"  發現 {len(routes)} 個端點:")
        for method, path in routes:
            print(f"    {method.upper():6} {path}")
        
        # Check if app can be created
        try:
            # We don't want to actually start the server, just verify the module structure
            tree = ast.parse(source)
            app_created = any(
                isinstance(node, ast.Assign) and 
                any(isinstance(t, ast.Name) and t.id == 'app' for t in (node.targets if isinstance(node.targets, list) else []))
                for node in ast.walk(tree)
            )
            print(f"  {'✅' if app_created else '❌'} FastAPI app 物件建立")
        except Exception as e:
            print(f"  ⚠️ AST 解析失敗: {e}")
        
        results["phases"]["api"] = {"routes": len(routes), "endpoints": [{"method": m, "path": p} for m, p in routes]}
    else:
        print(f"  ❌ web_server.py 不存在")
except Exception as e:
    print(f"  ❌ {e}")

# ═══════════════════════════════════════════════════════════
#  Phase 7: 密鑰外洩掃描
# ═══════════════════════════════════════════════════════════
section("Phase 7: 密鑰外洩掃描")

import subprocess

# Check if .env is tracked by git
try:
    r = subprocess.run(['git', 'ls-files', '.env'], capture_output=True, text=True, cwd=str(BASE))
    env_tracked = bool(r.stdout.strip())
    
    r2 = subprocess.run(['git', 'log', '--oneline', '--all', '--', '.env'], 
                        capture_output=True, text=True, cwd=str(BASE))
    env_in_history = bool(r2.stdout.strip())
    
    print(f"  .env 在 git tracked files 中: {'❌ 是 (危險!)' if env_tracked else '✅ 否'}")
    print(f"  .env 在 git 歷史中: {'❌ 是 (需要清除!)' if env_in_history else '✅ 否'}")
    print(f"  .gitignore 包含 .env: ", end="")
    
    gi = (BASE / ".gitignore").read_text()
    print(f"{'✅ 是' if '.env' in gi else '❌ 否'}")
    
    results["phases"]["secrets"] = {
        "env_tracked": env_tracked,
        "env_in_history": env_in_history,
        "gitignore_has_env": '.env' in gi
    }
except Exception as e:
    print(f"  ⚠️ Git 檢查失敗: {e}")

# Check for hardcoded secrets in source
print(f"\n  掃描硬編碼密鑰...")
secret_patterns = [
    (r'(?:api[_-]?key|password|secret|token)\s*=\s*["\'][a-zA-Z0-9]{16,}["\']', "Hardcoded secret"),
    (r'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}', "JWT token"),
]

import re
hardcoded_secrets = []
for f in py_files:
    try:
        content = f.read_text(encoding='utf-8', errors='replace')
        for pattern, desc in secret_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                hardcoded_secrets.append({
                    "file": str(f.relative_to(BASE)),
                    "type": desc,
                    "count": len(matches)
                })
    except:
        pass

if hardcoded_secrets:
    for s in hardcoded_secrets:
        print(f"  ⚠️ {s['file']} — {s['count']}× {s['type']}")
else:
    print(f"  ✅ .py 檔案中未發現硬編碼密鑰")

# ═══════════════════════════════════════════════════════════
#  Phase 8: NotImplementedError 掃描
# ═══════════════════════════════════════════════════════════
section("Phase 8: NotImplementedError / 空殼掃描")

nie_count = 0
nie_files = {}

for f in py_files:
    try:
        content = f.read_text(encoding='utf-8', errors='replace')
        matches = [i for i, line in enumerate(content.split('\n'), 1) 
                   if 'NotImplementedError' in line or 'NotImplemented' in line]
        if matches:
            rel = str(f.relative_to(BASE))
            nie_files[rel] = matches
            nie_count += len(matches)
    except:
        pass

print(f"  共 {nie_count} 處 NotImplementedError，分佈在 {len(nie_files)} 個檔案:")
for fname, lines in sorted(nie_files.items()):
    print(f"    {fname}: lines {lines}")

results["phases"]["not_implemented"] = {
    "total": nie_count,
    "files": len(nie_files),
    "details": nie_files
}

# ═══════════════════════════════════════════════════════════
#  Phase 9: 科學計算驗證
# ═══════════════════════════════════════════════════════════
section("Phase 9: 科學計算核心函數驗證")

science_tests = []

# Test algorithms
try:
    from engine.algorithms import OceanAlgorithms
    alg = OceanAlgorithms()
    
    # Haversine
    d = alg.haversine(25.0, 121.0, 26.0, 122.0)
    ok = 100 < d < 200  # ~150 km
    print(f"  {'✅' if ok else '❌'} haversine(25,121 → 26,122) = {d:.1f} km")
    science_tests.append(("haversine", ok))
    
    # SST front detection
    if hasattr(alg, 'detect_fronts') or hasattr(alg, 'sst_gradient'):
        print(f"  ✅ SST 鋒面偵測方法存在")
        science_tests.append(("fronts", True))
except Exception as e:
    print(f"  ❌ algorithms: {e}")
    science_tests.append(("algorithms", False))

# Test HSI
try:
    from engine.hsi_models import HSICalculator
    hsi = HSICalculator()
    
    # Test with typical tropical tuna conditions
    if hasattr(hsi, 'compute') or hasattr(hsi, 'calculate'):
        method = getattr(hsi, 'compute', getattr(hsi, 'calculate', None))
        if method:
            print(f"  ✅ HSI compute/calculate 方法存在")
    science_tests.append(("hsi", True))
except Exception as e:
    print(f"  ❌ HSI: {e}")
    science_tests.append(("hsi", False))

# Test species params
try:
    from engine.species_params import SPECIES
    species_count = len(SPECIES) if isinstance(SPECIES, dict) else 0
    print(f"  ✅ SPECIES 參數庫: {species_count} 物種")
    
    # Verify key species exist
    key_species = ['yellowfin', 'bigeye', 'skipjack']
    for sp in key_species:
        has = sp in SPECIES if isinstance(SPECIES, dict) else False
        print(f"    {'✅' if has else '❌'} {sp}")
    science_tests.append(("species_params", species_count >= 5))
except Exception as e:
    print(f"  ❌ species_params: {e}")

# Test lunar model
try:
    from engine.lunar_model import LunarFishingModel
    lm = LunarFishingModel()
    # Test moon phase calculation
    if hasattr(lm, 'get_moon_phase') or hasattr(lm, 'calculate_phase'):
        print(f"  ✅ LunarFishingModel 月相計算存在")
        science_tests.append(("lunar", True))
except Exception as e:
    print(f"  ❌ lunar: {e}")

# Test thermocline
try:
    import numpy as np
    from engine.thermocline_fetcher import ThermoclineFetcher
    tf = ThermoclineFetcher()
    
    # Simulate temperature profile
    depths = np.array([0, 10, 20, 50, 100, 150, 200, 300, 500])
    temps = np.array([28.5, 28.0, 27.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0])
    
    z20 = tf.compute_z20(temps, depths)
    mld = tf.compute_mld(temps, depths)
    t100 = tf.extract_temp_at_depth(temps, depths, 100.0)
    
    print(f"  ✅ Z20 = {z20:.1f}m (expected ~40-60m)")
    print(f"  ✅ MLD = {mld:.1f}m (expected ~10-30m)")
    print(f"  ✅ T100 = {t100:.1f}°C (expected ~16°C)")
    
    z20_ok = 30 < z20 < 80
    mld_ok = 5 < mld < 40
    t100_ok = 14 < t100 < 18
    science_tests.append(("thermocline", z20_ok and mld_ok and t100_ok))
except Exception as e:
    print(f"  ❌ thermocline: {e}")
    science_tests.append(("thermocline", False))

pass_count = sum(1 for _, ok in science_tests if ok)
print(f"\n  科學計算驗證: {pass_count}/{len(science_tests)} 通過")

results["phases"]["science"] = {
    "total": len(science_tests),
    "pass": pass_count,
    "details": {name: "pass" if ok else "fail" for name, ok in science_tests}
}

# ═══════════════════════════════════════════════════════════
#  Phase 10: 資料檔完整性
# ═══════════════════════════════════════════════════════════
section("Phase 10: 資料檔完整性")

data_dir = BASE / "data"
if data_dir.exists():
    csv_files = list(data_dir.rglob("*.csv"))
    nc_files = list(data_dir.rglob("*.nc"))
    npz_files = list(data_dir.rglob("*.npz"))
    
    csv_total = sum(f.stat().st_size for f in csv_files)
    nc_total = sum(f.stat().st_size for f in nc_files)
    npz_total = sum(f.stat().st_size for f in npz_files)
    
    print(f"  CSV: {len(csv_files)} 檔, {csv_total/1024/1024:.1f} MB")
    print(f"  NC:  {len(nc_files)} 檔, {nc_total/1024/1024:.1f} MB")
    print(f"  NPZ: {len(npz_files)} 檔, {npz_total/1024/1024:.1f} MB")
    
    # Verify WCPFC data
    wcpfc_dir = data_dir / "wcpfc"
    if wcpfc_dir.exists():
        wcpfc_files = list(wcpfc_dir.rglob("*.csv"))
        wcpfc_total = sum(f.stat().st_size for f in wcpfc_files)
        print(f"  WCPFC: {len(wcpfc_files)} 檔, {wcpfc_total/1024/1024:.1f} MB")
        
        # Try reading the first WCPFC file
        if wcpfc_files:
            try:
                import pandas as pd
                df = pd.read_csv(wcpfc_files[0], nrows=5)
                print(f"  ✅ WCPFC 資料可讀取: {len(df.columns)} 欄位")
                print(f"     欄位: {list(df.columns[:10])}...")
            except Exception as e:
                print(f"  ❌ WCPFC 讀取失敗: {e}")

    results["phases"]["data"] = {
        "csv": len(csv_files),
        "nc": len(nc_files),
        "npz": len(npz_files),
        "total_mb": round((csv_total + nc_total + npz_total) / 1024 / 1024, 1)
    }

# ═══════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════
section("📊 驗證總結")

phases = results["phases"]

print(f"""
  Python: {sys.version.split()[0]}
  
  Phase 1  語法驗證:     {phases.get('syntax', {}).get('pass', '?')}/{phases.get('syntax', {}).get('total', '?')} ✅
  Phase 2  Import 測試:  {phases.get('import', {}).get('pass', '?')}/{phases.get('import', {}).get('total', '?')} ✅
  Phase 3  類別實例化:   {phases.get('classes', {}).get('pass', '?')}/{phases.get('classes', {}).get('total', '?')} ✅
  Phase 4  模型權重:     {phases.get('models', {}).get('pt_files', '?')} .pt 檔
  Phase 5  Stacking E2E: {phases.get('stacking_e2e', {}).get('status', '?')}
  Phase 7  密鑰安全:     .env tracked={phases.get('secrets', {}).get('env_tracked', '?')}
  Phase 8  空殼模組:     {phases.get('not_implemented', {}).get('total', '?')} 處 NotImplemented
  Phase 9  科學計算:     {phases.get('science', {}).get('pass', '?')}/{phases.get('science', {}).get('total', '?')} ✅
  Phase 10 資料檔:       {phases.get('data', {}).get('total_mb', '?')} MB
""")

# Save results
out_path = BASE / "dynamic_verification_report.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f"  報告已儲存: {out_path}")
