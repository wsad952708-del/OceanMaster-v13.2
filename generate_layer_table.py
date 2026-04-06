import json
import os

with open(r'c:\Users\user\Desktop\好像快好了\scan_result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tree = data.get('tree', {})

def count_files(node):
    if not isinstance(node, dict):
        return 0
    count = len(node.get('_files', []))
    for k, v in node.items():
        if k != '_files' and isinstance(v, dict):
            count += count_files(v)
    return count

layers = {
    "Core Engine & Pipeline": ["engine", "pipeline", "ml_system"],
    "Models & Database": ["models", "edge_models", "fm_checkpoints", "trained_models", "rag_database"],
    "API & Frontend": ["api", "web", "static"],
    "Data & Fetchers": ["data", "data_fetcher_external"],
    "Infra & Deployment": ["deploy", "scripts", "scheduler"],
    "Tests & Docs": ["tests", "docs"],
    "Cache & Output": ["cache", "output", "venv"],
    "Root & Configs": ["_files"]
}

risks = {
    "Core Engine & Pipeline": "🔴 CRITICAL",
    "Models & Database": "🟡 WARNING",
    "API & Frontend": "🟡 WARNING",
    "Data & Fetchers": "🟡 WARNING",
    "Infra & Deployment": "🔴 CRITICAL",
    "Tests & Docs": "🟢 INFO",
    "Cache & Output": "🟢 INFO",
    "Root & Configs": "🔴 CRITICAL"
}

results = []

for layer_name, paths in layers.items():
    layer_count = 0
    actual_paths = []
    for path in paths:
        if path == "_files":
            layer_count += len(tree.get("_files", []))
            actual_paths.append("/")
        elif path in tree:
            layer_count += count_files(tree[path])
            actual_paths.append(f"{path}/")
    
    if layer_count > 0:
        results.append(f"| {layer_name} | `{', '.join(actual_paths)}` | {layer_count} | {risks[layer_name]} |")

with open(r'c:\Users\user\Desktop\好像快好了\layer_table.md', 'w', encoding='utf-8') as f:
    f.write("| Layer | Path | File Count | Est. Risk |\n")
    f.write("|-------|------|------------|-----------|\n")
    for r in results:
        f.write(r + "\n")
