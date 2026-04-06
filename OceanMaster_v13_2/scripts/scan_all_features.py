"""Scan ALL engine modules — extract every public function/class and what it does"""
import sys, os, importlib, inspect
os.chdir(r'c:\Users\user\Desktop\好像快好了\OceanMaster_v13_2')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import glob

engine_files = sorted(glob.glob('engine/*.py') + glob.glob('engine/ml/*.py'))
engine_files = [f for f in engine_files if '__pycache__' not in f and '__init__' not in f]

print(f"Total engine files: {len(engine_files)}\n")

for fp in engine_files:
    mod_path = fp.replace('\\','/').replace('/','.').replace('.py','')
    try:
        mod = importlib.import_module(mod_path)
        # Get public classes and functions
        classes = []
        functions = []
        for name, obj in inspect.getmembers(mod):
            if name.startswith('_'): continue
            if inspect.isclass(obj) and obj.__module__ == mod.__name__:
                methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]
                doc = (obj.__doc__ or '').strip().split('\n')[0][:80] if obj.__doc__ else ''
                classes.append((name, methods[:5], doc))
            elif inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                sig = str(inspect.signature(obj))[:60]
                doc = (obj.__doc__ or '').strip().split('\n')[0][:60] if obj.__doc__ else ''
                functions.append((name, sig, doc))
        
        if classes or functions:
            print(f"{'─'*60}")
            print(f"📄 {fp}")
            for cname, methods, doc in classes:
                print(f"  class {cname}: {doc}")
                if methods: print(f"    methods: {', '.join(methods)}")
            for fname, sig, doc in functions:
                print(f"  def {fname}{sig}")
                if doc: print(f"    → {doc}")
    except Exception as e:
        print(f"{'─'*60}")
        print(f"📄 {fp} — IMPORT ERROR: {str(e)[:60]}")
