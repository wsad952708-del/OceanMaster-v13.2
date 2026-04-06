"""Final deep verification - fill all gaps"""
import os, sys, re, ast, importlib
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
EXCLUDE = {'venv', '__pycache__', '.git', 'node_modules'}

def get_py_files():
    files = list(BASE.rglob('*.py'))
    return [f for f in files if not any(x in str(f) for x in ['venv', '__pycache__', '.git'])]

py_files = get_py_files()

# === 1. Dependency graph: who imports whom ===
print('=== 1. DEPENDENCY GRAPH ===')
import_map = defaultdict(set)  # module -> set of importers

for f in py_files:
    try:
        content = f.read_text(encoding='utf-8', errors='replace')
        rel = str(f.relative_to(BASE))
        
        # Find all "from engine.X import" and "import engine.X"
        for m in re.findall(r'from\s+(engine\.[a-zA-Z_.]+)\s+import', content):
            base_mod = m.split('.')[0] + '.' + m.split('.')[1] if len(m.split('.')) > 1 else m
            import_map[base_mod].add(rel)
        for m in re.findall(r'import\s+(engine\.[a-zA-Z_.]+)', content):
            base_mod = m.split('.')[0] + '.' + m.split('.')[1] if len(m.split('.')) > 1 else m
            import_map[base_mod].add(rel)
    except:
        pass

# Show top imported modules
sorted_mods = sorted(import_map.items(), key=lambda x: len(x[1]), reverse=True)
for mod, importers in sorted_mods[:25]:
    print('  %s -> %d importers' % (mod, len(importers)))
    for imp in sorted(importers)[:5]:
        print('    <- %s' % imp)
    if len(importers) > 5:
        print('    ... and %d more' % (len(importers) - 5))

# === 2. Haversine count ===
print('\n=== 2. HAVERSINE DUPLICATES ===')
haversine_locs = []
for f in py_files:
    try:
        content = f.read_text(encoding='utf-8', errors='replace')
        for i, line in enumerate(content.split('\n'), 1):
            if 'def haversine' in line.lower() or 'def _haversine' in line.lower():
                haversine_locs.append((str(f.relative_to(BASE)), i, line.strip()))
    except:
        pass
print('  Haversine function definitions: %d' % len(haversine_locs))
for loc in haversine_locs:
    print('    %s:%d  %s' % loc)

# Also check haversine usage (not just definitions)
haversine_usage = 0
for f in py_files:
    try:
        content = f.read_text(encoding='utf-8', errors='replace')
        count = len(re.findall(r'haversine', content, re.IGNORECASE))
        if count > 0 and 'def haversine' not in content.lower():
            haversine_usage += 1
    except:
        pass
print('  Files using haversine (not defining): %d' % haversine_usage)

# === 3. Isolated module detection ===
print('\n=== 3. ISOLATED MODULES ===')
engine_dir = BASE / 'engine'
engine_mods = []
for f in engine_dir.rglob('*.py'):
    if '__pycache__' in str(f) or f.name == '__init__.py':
        continue
    engine_mods.append(f)

for f in sorted(engine_mods, key=lambda x: x.stat().st_size, reverse=True):
    mod_stem = f.stem
    # Count how many OTHER files reference this module
    ref_count = 0
    ref_files = []
    for pf in py_files:
        if pf == f:
            continue
        try:
            content = pf.read_text(encoding='utf-8', errors='replace')
            if mod_stem in content:
                ref_count += 1
                ref_files.append(str(pf.relative_to(BASE)))
        except:
            pass
    
    if ref_count <= 2:
        sz = f.stat().st_size / 1024
        tag = 'ORPHAN' if ref_count == 0 else ('NEAR-ORPHAN' if ref_count == 1 else 'LOW-REF')
        print('  [%s] %s (%.1f KB) -> %d refs: %s' % (tag, f.relative_to(BASE), sz, ref_count, ref_files[:3]))

# === 4. NC files everywhere ===
print('\n=== 4. ALL .NC FILES (entire project) ===')
nc_files = [f for f in BASE.rglob('*.nc') if 'venv' not in str(f)]
for f in nc_files:
    print('  %s: %.1f MB' % (f.relative_to(BASE), f.stat().st_size/1024/1024))
print('  Total: %d NC files' % len(nc_files))

# === 5. Feature dimensions ===
print('\n=== 5. FEATURE ENGINEER DIMENSIONS ===')
try:
    se_path = BASE / 'engine' / 'ml' / 'stacking_ensemble.py'
    content = se_path.read_text(encoding='utf-8', errors='replace')
    
    # Find extract_features method and look for feature names
    in_extract = False
    feature_lines = []
    for i, line in enumerate(content.split('\n'), 1):
        if 'def extract_features' in line:
            in_extract = True
        if in_extract:
            # Look for feature assignment patterns
            if any(x in line for x in ['features[', "features['", 'features["', 'append(', '.extend(']):
                feature_lines.append((i, line.strip()))
            if in_extract and line.strip().startswith('def ') and 'extract_features' not in line:
                break
    
    print('  Feature assignment lines in extract_features():')
    for ln, txt in feature_lines[:50]:
        print('    L%d: %s' % (ln, txt[:100]))
    
    # Also find FEATURE_NAMES or similar constants
    for i, line in enumerate(content.split('\n'), 1):
        if 'FEATURE_NAMES' in line or 'feature_names' in line:
            print('  L%d: %s' % (i, line.strip()[:120]))
    
    # Find backward compat code
    for i, line in enumerate(content.split('\n'), 1):
        if 'n_model_features' in line or 'shape[1]' in line:
            print('  COMPAT L%d: %s' % (i, line.strip()[:120]))
            
except Exception as e:
    print('  ERROR: %s' % e)

# === 6. Main entry point ===
print('\n=== 6. MAIN ENTRY POINT ===')
main_path = BASE / 'main_v10_3.py'
if main_path.exists():
    sz = main_path.stat().st_size / 1024
    lines = len(main_path.read_text(encoding='utf-8', errors='replace').split('\n'))
    print('  main_v10_3.py: %.1f KB, %d lines' % (sz, lines))
    
    # Find what it imports from engine
    content = main_path.read_text(encoding='utf-8', errors='replace')
    engine_imports = re.findall(r'from\s+(engine\.[a-zA-Z_.]+)\s+import', content)
    print('  Imports from engine: %d modules' % len(engine_imports))
    for imp in sorted(set(engine_imports)):
        print('    %s' % imp)

# === 7. web/ and static/ inventory ===
print('\n=== 7. WEB FRONTEND INVENTORY ===')
for d_name in ['web', 'static']:
    d = BASE / d_name
    if d.exists():
        for f in sorted(d.rglob('*')):
            if f.is_file():
                sz = f.stat().st_size / 1024
                print('  %s: %.1f KB' % (f.relative_to(BASE), sz))

# === 8. data_fetcher_v2.py deep check ===
print('\n=== 8. DATA_FETCHER_V2.PY ===')
df_path = BASE / 'engine' / 'data_fetcher_v2.py'
if df_path.exists():
    content = df_path.read_text(encoding='utf-8', errors='replace')
    lines = len(content.split('\n'))
    sz = df_path.stat().st_size / 1024
    classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
    funcs = re.findall(r'^\s*def\s+(\w+)', content, re.MULTILINE)
    print('  Size: %.1f KB, %d lines' % (sz, lines))
    print('  Classes: %s' % classes)
    print('  Functions (%d): %s' % (len(funcs), funcs[:20]))

# === 9. Architecture validation - trace data flow ===
print('\n=== 9. DATA FLOW TRACE ===')
# Check which modules import from which layer
layers = {
    'input': ['data_fetcher', 'weather_fetcher', 'wave_fetcher', 'gebco', 'dissolved_oxygen', 
              'typhoon_tracker', 'gfw_data', 'vessel_lights', 'rainfall', 'mur_sst', 'cmems'],
    'feature': ['stacking_ensemble', 'lagrangian', 'kuroshio', 'lunar', 'algorithms', 'okubo'],
    'ml': ['unet', 'convlstm', 'transfish', 'srgan', 'dl_models', 'stacking'],
    'fusion': ['hsi_models', 'ai_fusion', 'safety_checker', 'shap_explainer', 'route_planner'],
    'output': ['dashboard', 'kml_generator', 'web_server', 'text_briefing', 'html_map']
}

for layer_name, keywords in layers.items():
    matching = []
    for f in engine_mods:
        if any(kw in f.stem for kw in keywords):
            matching.append(f.stem)
    print('  %s layer: %d modules (%s)' % (layer_name, len(matching), matching[:8]))

print('\n=== DONE ===')
