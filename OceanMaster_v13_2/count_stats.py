import os
import ast

directory = r"C:\Users\user\Desktop\好像快好了\OceanMaster_v13_2"

total_lines = 0
effective_lines = 0
class_count = 0
function_count = 0
todo_count = 0
not_implemented_count = 0

for root, dirs, files in os.walk(directory):
    if 'venv' in root or '.git' in root or '__pycache__' in root or 'node_modules' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for line in lines:
                        total_lines += 1
                        stripped = line.strip()
                        # Exclude empty lines and comment lines/docstrings heuristically
                        if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                            effective_lines += 1
                        if 'TODO' in line:
                            todo_count += 1
                        if 'NotImplementedError' in line:
                            not_implemented_count += 1
                            
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                class_count += 1
                            elif isinstance(node, ast.FunctionDef):
                                function_count += 1
                    except SyntaxError:
                        pass
            except Exception as e:
                pass

print(f"Total Python Lines: {total_lines}")
print(f"Effective Python LOC: {effective_lines}")
print(f"Total Custom Classes: {class_count}")
print(f"Total Custom Functions: {function_count}")
print(f"TODO Count: {todo_count}")
print(f"NotImplementedError Count: {not_implemented_count}")
