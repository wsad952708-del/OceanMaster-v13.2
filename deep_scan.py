import os
import ast
import json

def deep_scan(root_dir):
    stats = {
        "total_py_files": 0,
        "total_lines_of_code": 0,
        "total_classes": 0,
        "total_functions": 0,
        "dl_models": 0,    # nn.Module or similar
        "ml_models": 0,    # sklearn/xgboost imports or usages
        "api_endpoints": 0, # @app. route or similar
        "fetchers": 0
    }

    ignore_dirs = {'.git', '__pycache__', 'venv', 'node_modules', '.idea', '.vscode'}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for f in filenames:
            if f.endswith('.py'):
                stats["total_py_files"] += 1
                filepath = os.path.join(dirpath, f)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                        lines = content.split('\n')
                        stats["total_lines_of_code"] += len([l for l in lines if l.strip() and not l.strip().startswith('#')])
                        
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                stats["total_classes"] += 1
                                # crude check for DL models
                                for base in node.bases:
                                    if isinstance(base, ast.Attribute) and base.attr == 'Module':
                                        stats["dl_models"] += 1
                                    elif isinstance(base, ast.Name) and base.id == 'Module':
                                        stats["dl_models"] += 1
                            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                                stats["total_functions"] += 1
                                # crude check for API endpoints
                                for decorator in node.decorator_list:
                                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                                        if decorator.func.attr in ['get', 'post', 'put', 'delete']:
                                            stats["api_endpoints"] += 1
                            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                                # check for ML models
                                module_name = getattr(node, 'module', '') or ''
                                if any(x in module_name for x in ['sklearn', 'xgboost', 'lightgbm', 'catboost']):
                                    stats["ml_models"] += 1
                                    
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")
                    
    with open('deep_stats.json', 'w', encoding='utf-8') as out:
        json.dump(stats, out, indent=2)

if __name__ == "__main__":
    deep_scan("C:/Users/user/Desktop/好像快好了/OceanMaster_v13_2")
