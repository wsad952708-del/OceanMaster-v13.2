"""Deep scan for accurate tech doc metrics"""
import os, ast, json, sys, re, importlib
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent
EXCLUDE = {'venv', '.venv', '__pycache__', '.git', 'node_modules'}
sys.path.insert(0, str(BASE))

# === 1. File counts by extension ===
ext_count = defaultdict(lambda: {'count': 0, 'size': 0})
total_files = 0
total_dirs = 0
for root, dirs, files in os.walk(BASE):
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    total_dirs += len(dirs)
    for f in files:
        fp = Path(root) / f
        try:
            sz = fp.stat().st_size
        except:
            sz = 0
        ext = fp.suffix.lower()
        ext_count[ext]['count'] += 1
        ext_count[ext]['size'] += sz
        total_files += 1

print('=== FILE COUNTS ===')
print('Total files:', total_files)
print('Total dirs:', total_dirs)
for ext in sorted(ext_count.keys(), key=lambda x: ext_count[x]['size'], reverse=True)[:25]:
    d = ext_count[ext]
    print('  %-10s %5d files  %8.1f MB' % (ext, d['count'], d['size']/1024/1024))

# === 2. Python files ===
py_files = list(BASE.rglob('*.py'))
py_files = [f for f in py_files if not any(x in str(f) for x in ['venv', '__pycache__', '.git'])]

print('\n=== PYTHON FILES: %d ===' % len(py_files))
total_lines = 0
for f in sorted(py_files):
    try:
        lines = len(f.read_text(encoding='utf-8', errors='replace').split('\n'))
        total_lines += lines
    except:
        lines = 0
print('Total .py lines:', total_lines)

# === 3. engine/ breakdown ===
engine_dir = BASE / 'engine'
engine_py = [f for f in py_files if str(f).startswith(str(engine_dir))]
engine_init = [f for f in engine_py if f.name == '__init__.py']
engine_real = [f for f in engine_py if f.name != '__init__.py']
print('\n=== ENGINE ===')
print('Total .py in engine/: %d (init: %d, real: %d)' % (len(engine_py), len(engine_init), len(engine_real)))

subdirs = defaultdict(list)
for f in engine_py:
    rel = f.relative_to(engine_dir)
    parts = rel.parts
    if len(parts) == 1:
        subdirs['ROOT'].append(f)
    else:
        subdirs[parts[0]].append(f)

for sd in sorted(subdirs.keys()):
    files = subdirs[sd]
    real = [f for f in files if f.name != '__init__.py']
    total_kb = sum(f.stat().st_size for f in files) / 1024
    total_l = sum(len(f.read_text(encoding='utf-8', errors='replace').split('\n')) for f in files)
    print('  %-20s %3d real .py  %7.1f KB  %6d lines' % (sd, len(real), total_kb, total_l))

# engine ROOT top files by size
print('\n=== ENGINE ROOT TOP 30 (by size) ===')
root_files = [f for f in engine_py if f.parent == engine_dir and f.name != '__init__.py']
root_files.sort(key=lambda f: f.stat().st_size, reverse=True)
for f in root_files[:30]:
    sz = f.stat().st_size / 1024
    lines = len(f.read_text(encoding='utf-8', errors='replace').split('\n'))
    print('  %-40s %6.1f KB  %5d lines' % (f.name, sz, lines))

# === 4. Data files ===
print('\n=== DATA FILES ===')
data_dir = BASE / 'data'
for ext in ['.csv', '.nc', '.npz', '.npy']:
    files = list(data_dir.rglob('*' + ext)) if data_dir.exists() else []
    total_sz = sum(f.stat().st_size for f in files)
    print('  %-6s %4d files  %8.1f MB' % (ext, len(files), total_sz/1024/1024))

# WCPFC
import pandas as pd
wcpfc = sorted((data_dir / 'wcpfc').rglob('*.csv')) if (data_dir / 'wcpfc').exists() else []
wcpfc_rows = 0
print('\n=== WCPFC DATA ===')
for f in wcpfc:
    try:
        df = pd.read_csv(f)
        wcpfc_rows += len(df)
        print('  %s: %s rows x %d cols -- %s' % (f.name, format(len(df), ','), len(df.columns), list(df.columns[:6])))
    except Exception as e:
        print('  %s: ERROR %s' % (f.name, e))
print('  TOTAL: %s rows' % format(wcpfc_rows, ','))

# === 5. Model weights ===
print('\n=== MODEL WEIGHTS ===')
pt_files = [f for f in BASE.rglob('*.pt') if 'venv' not in str(f)]
for f in pt_files:
    print('  %s: %.1f MB' % (f.relative_to(BASE), f.stat().st_size/1024/1024))

# === 6. Tests ===
tests_dir = BASE / 'tests'
test_files = [f for f in tests_dir.glob('*.py') if f.name != '__init__.py'] if tests_dir.exists() else []
print('\n=== TESTS: %d files ===' % len(test_files))
for f in sorted(test_files):
    lines = len(f.read_text(encoding='utf-8', errors='replace').split('\n'))
    print('  %s: %d lines' % (f.name, lines))

# === 7. FastAPI endpoints ===
print('\n=== FASTAPI ENDPOINTS ===')
ws_path = BASE / 'web_server.py'
if ws_path.exists():
    source = ws_path.read_text(encoding='utf-8', errors='replace')
    ws_lines = len(source.split('\n'))
    routes = re.findall(r'@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']', source)
    print('web_server.py: %d lines' % ws_lines)
    for method, path in routes:
        print('  %s %s' % (method.upper(), path))
    print('  Total: %d endpoints' % len(routes))

# === 8. Web frontend ===
print('\n=== WEB FRONTEND ===')
web_dir = BASE / 'web'
static_dir = BASE / 'static'
for d in [web_dir, static_dir]:
    if d.exists():
        for f in d.rglob('*'):
            if f.is_file():
                print('  %s: %.1f KB' % (f.relative_to(BASE), f.stat().st_size/1024))

# === 9. Classes and functions in key modules ===
print('\n=== KEY MODULE API ===')
key_modules = [
    'engine.algorithms',
    'engine.hsi_models',
    'engine.ai_fusion',
    'engine.safety_checker',
    'engine.gebco_features',
    'engine.lunar_model',
    'engine.kuroshio_engine',
    'engine.typhoon_tracker',
    'engine.shap_explainer',
    'engine.ml.stacking_ensemble',
    'engine.species_params',
    'engine.wave_fetcher',
    'engine.weather_fetcher',
]

for mod_name in key_modules:
    try:
        mod = importlib.import_module(mod_name)
        # Get classes
        classes = []
        functions = []
        constants = []
        for name in sorted(dir(mod)):
            if name.startswith('_'):
                continue
            obj = getattr(mod, name)
            if isinstance(obj, type):
                methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]
                classes.append((name, methods))
            elif callable(obj) and not isinstance(obj, type):
                if name[0].islower():
                    functions.append(name)
            elif name[0].isupper() and not callable(obj):
                constants.append(name)
        
        print('\n  [%s]' % mod_name)
        for cname, methods in classes:
            print('    class %s: %s' % (cname, methods))
        if functions:
            print('    functions: %s' % functions)
        if constants:
            print('    constants: %s' % constants[:10])
    except Exception as e:
        print('\n  [%s] IMPORT FAILED: %s' % (mod_name, e))

# === 10. NotImplementedError scan ===
print('\n=== NotImplementedError SCAN ===')
nie_total = 0
nie_files = {}
for f in py_files:
    try:
        content = f.read_text(encoding='utf-8', errors='replace')
        matches = []
        for i, line in enumerate(content.split('\n'), 1):
            if 'NotImplementedError' in line and 'count' not in str(f.name) and 'dynamic_ver' not in str(f.name) and 'verify' not in str(f.name):
                matches.append(i)
        if matches:
            rel = str(f.relative_to(BASE))
            nie_files[rel] = matches
            nie_total += len(matches)
    except:
        pass

print('Total: %d occurrences in %d files' % (nie_total, len(nie_files)))
for fname, lines in sorted(nie_files.items()):
    print('  %s: %d times (lines %s)' % (fname, len(lines), lines))

# === 11. Syntax errors ===
print('\n=== SYNTAX ERRORS ===')
for f in py_files:
    try:
        source = f.read_text(encoding='utf-8', errors='replace')
        ast.parse(source)
    except SyntaxError as e:
        print('  %s:%d -- %s' % (f.relative_to(BASE), e.lineno, e.msg))

# === 12. Import success rate ===
print('\n=== ENGINE IMPORT TEST ===')
ok_count = 0
fail_count = 0
fail_list = []
for f in sorted(engine_real):
    rel = f.relative_to(BASE).with_suffix('')
    mod_name = str(rel).replace(os.sep, '.')
    try:
        importlib.import_module(mod_name)
        ok_count += 1
    except Exception as e:
        fail_count += 1
        fail_list.append((mod_name, type(e).__name__, str(e)[:80]))

print('Pass: %d/%d (%.1f%%)' % (ok_count, ok_count + fail_count, ok_count/(ok_count+fail_count)*100))
print('Failures:')
for mod, etype, msg in fail_list:
    print('  %s: %s: %s' % (mod, etype, msg))

# === 13. Deploy files ===
print('\n=== DEPLOY FILES ===')
deploy_files = ['Dockerfile', 'Dockerfile.production', 'docker-compose.yml', 
                'docker-compose.production.yml', 'requirements.txt', '.env.example']
for fname in deploy_files:
    fp = BASE / fname
    if fp.exists():
        print('  %s: %.1f KB' % (fname, fp.stat().st_size/1024))
    else:
        print('  %s: MISSING' % fname)

deploy_dir = BASE / 'deploy'
if deploy_dir.exists():
    for f in deploy_dir.iterdir():
        if f.is_file():
            print('  deploy/%s: %.1f KB' % (f.name, f.stat().st_size/1024))

# === 14. Total size ===
total_size = 0
for root, dirs, files in os.walk(BASE):
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    for f in files:
        try:
            total_size += (Path(root)/f).stat().st_size
        except:
            pass
print('\n=== TOTAL PROJECT SIZE: %.2f GB ===' % (total_size/1024/1024/1024))

# === 15. requirements.txt packages ===
print('\n=== REQUIREMENTS.TXT ===')
req = (BASE / 'requirements.txt').read_text()
for line in req.strip().split('\n'):
    line = line.strip()
    if line and not line.startswith('#'):
        print('  %s' % line)
