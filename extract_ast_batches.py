import os, ast

ROOT = r"C:\Users\user\Desktop\好像快好了"
CHUNK_SIZE = 50  # 每次處理50個py檔案

py_files = []
for root, dirs, files in os.walk(ROOT):
    # 排除非專案本身的目錄，減少雜訊與避免解析虛擬環境
    dirs[:] = [d for d in dirs if d not in ['.git', 'venv', '__pycache__', '.pytest_cache']]
    for f in files:
        if f.endswith(".py"):
            py_files.append(os.path.join(root, f))

# 分批處理
for batch_num, i in enumerate(range(0, len(py_files), CHUNK_SIZE)):
    batch = py_files[i:i+CHUNK_SIZE]
    output = f"=== 批次 {batch_num+1} / {len(py_files)//CHUNK_SIZE+1} ===\n\n"
    
    for fpath in batch:
        rel = os.path.relpath(fpath, ROOT)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            
            # 用AST解析，只取函數/類別名稱，不取完整代碼
            tree = ast.parse(code)
            classes = [n.name for n in ast.walk(tree) 
                      if isinstance(n, ast.ClassDef)]
            funcs = [n.name for n in ast.walk(tree) 
                    if isinstance(n, ast.FunctionDef)]
            imports = [ast.dump(n)[:60] for n in ast.walk(tree) 
                      if isinstance(n, (ast.Import, ast.ImportFrom))][:10]
            
            output += f"📄 {rel}\n"
            output += f"  Classes: {classes}\n"
            output += f"  Functions: {funcs[:20]}\n"
            output += f"  Imports: {imports}\n\n"
        except Exception as e:
            output += f"📄 {rel} [無法解析: {e}]\n\n"
    
    with open(os.path.join(ROOT, f"batch_{batch_num+1:03d}.txt"), "w", encoding="utf-8") as f:
        f.write(output)

print(f"[OK] 共生成 {batch_num+1} 個批次檔案")
