import os
import json
from pathlib import Path

ROOT = r"C:\Users\user\Desktop\好像快好了"
output_lines = []
ai_keywords = ["DL","RL","FL","FM","agent","model","train","predict",
               "dataset","neural","transformer","llm","gpt","bert",
               "crawler","spider","scrape","api","fetch"]

summary = {
    "total_files": 0,
    "total_folders": 0,
    "file_types": {},
    "ai_related_files": [],
    "folder_tree": []
}

for root, dirs, files in os.walk(ROOT):
    level = root.replace(ROOT, '').count(os.sep)
    indent = '  ' * level
    folder_name = os.path.basename(root)
    summary["total_folders"] += 1
    summary["folder_tree"].append(f"{indent}📁 {folder_name}/")
    
    for file in files:
        summary["total_files"] += 1
        ext = Path(file).suffix.lower()
        summary["file_types"][ext] = summary["file_types"].get(ext, 0) + 1
        
        # 找出跟AI相關的檔案
        if any(k in file.lower() for k in ai_keywords):
            rel_path = os.path.relpath(os.path.join(root, file), ROOT)
            summary["ai_related_files"].append(rel_path)
        
        if level <= 2:  # 只顯示前兩層的檔案
            summary["folder_tree"].append(f"{indent}  📄 {file}")

# 輸出結果
with open("folder_map.txt", "w", encoding="utf-8") as f:
    f.write("=== 資料夾地圖 ===\n")
    f.write(f"總檔案數: {summary['total_files']}\n")
    f.write(f"總資料夾數: {summary['total_folders']}\n\n")
    
    f.write("=== 檔案類型統計 ===\n")
    for ext, count in sorted(summary["file_types"].items(), 
                              key=lambda x: -x[1])[:30]:
        f.write(f"  {ext or '無副檔名'}: {count} 個\n")
    
    f.write("\n=== AI相關檔案 ===\n")
    for p in summary["ai_related_files"][:200]:
        f.write(f"  {p}\n")
    
    f.write("\n=== 目錄結構（前兩層）===\n")
    f.write("\n".join(summary["folder_tree"]))

print("[OK] folder_map.txt 已生成！")
