import os
import sys
import json
from collections import Counter

def scan_project(root_path):
    stats = {
        "total_files": 0,
        "total_folders": 0,
        "extensions": Counter(),
        "tree": {},
        "all_files": []
    }
    
    ignore_dirs = {'.git', '__pycache__', 'venv', 'node_modules', '.idea', '.vscode'}
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter ignored directories
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        
        rel_path = os.path.relpath(dirpath, root_path)
        depth = rel_path.count(os.sep) if rel_path != '.' else 0
        
        if rel_path != '.':
            stats["total_folders"] += 1
            
        for f in filenames:
            stats["total_files"] += 1
            ext = os.path.splitext(f)[1].lower()
            if not ext:
                ext = 'no_extension'
            stats["extensions"][ext] += 1
            
            f_rel = os.path.join(rel_path, f) if rel_path != '.' else f
            stats["all_files"].append(f_rel)
            
            if depth < 3:
                # build tree visualization structure (simple)
                parts = f_rel.split(os.sep)
                current = stats["tree"]
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        if "_files" not in current:
                            current["_files"] = []
                        current["_files"].append(part)
                    else:
                        if part not in current:
                            current[part] = {}
                        current = current[part]

    stats["extensions"] = dict(stats["extensions"])
    
    with open('scan_result.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        
    print("Scan complete. Results saved to scan_result.json")

if __name__ == "__main__":
    scan_project(sys.argv[1] if len(sys.argv) > 1 else ".")
