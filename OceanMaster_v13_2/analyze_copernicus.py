# /path/to/project/analyze_copernicus.py
"""Copernicus / CMEMS 關鍵字掃描管線（DES-002 修復版）。

漁業科學意義
============
本模組遞迴掃描 OceanMaster 專案檔案，搜尋與 Copernicus Marine
Environment Monitoring Service (CMEMS) 相關的關鍵字，用以：

1. 盤點所有引用 ``copernicusmarine``、``copernicus``、``cmems``、
   ``netcdf`` 等 API 的程式碼與文件。
2. 為 WCPFC / IOTC 資料提交合規審計提供完整的 processing lineage——
   確認哪些模組負責下載、解析 CMEMS NetCDF 產品。
3. 偵測假陰性：``scanned_count + error_count == total_eligible_files``
   確保沒有檔案被靜默跳過。

科學約束
--------
- FDC-001: 禁止靜默吞噬例外；所有 ``except`` 區塊皆寫入結構化日誌。
- FDC-002: 禁止硬編碼絕對路徑；使用 ``OCEANMASTER_ROOT`` 環境變數。
- CMEMS NetCDF 讀取失敗時須區分 timeout / corrupt / version_mismatch。
- 葉綠素規則 CHL_RANGE: ``0.01 <= chl <= 100.0 mg/m³``。
"""
from __future__ import annotations

import errno
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 共用日誌模組（DES-007 基礎設施）
# ---------------------------------------------------------------------------
from oceanmaster_logging import log_structured_error, logger, setup_logger

# fallback: 若 oceanmaster_logging 不可用，至少有標準 logging
# logger = logging.getLogger(__name__)  # uncomment as fallback

# ---------------------------------------------------------------------------
# 常數與組態
# ---------------------------------------------------------------------------
# DES-002: 將版本化目錄名稱外移至環境變數，消除硬編碼版本字串
_DEFAULT_ROOT_DIR_NAME: str = os.environ.get(
    "OCEANMASTER_DIR_NAME", "OceanMaster_v13_2"
)
_DEFAULT_ROOT: str = os.path.join(os.path.expanduser("~"), _DEFAULT_ROOT_DIR_NAME)
OCEANMASTER_ROOT: str = os.environ.get("OCEANMASTER_ROOT", _DEFAULT_ROOT)

# ---------------------------------------------------------------------------
# 審計計數器（FDC-001 假陰性防護）
# ---------------------------------------------------------------------------
error_count: int = 0
scanned_count: int = 0
skipped_count: int = 0

# Fail-fast：若 OCEANMASTER_ROOT 未明確設定，且 fallback 路徑不存在，
# 立即失敗以防止靜默掃描空目錄後回傳 0 matches 的假陰性。
if "OCEANMASTER_ROOT" not in os.environ:
    logger.warning(
        json.dumps(
            {
                "event": "env_var_missing",
                "variable": "OCEANMASTER_ROOT",
                "fallback": _DEFAULT_ROOT,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    if not Path(_DEFAULT_ROOT).is_dir():
        raise FileNotFoundError(
            f"OCEANMASTER_ROOT 未設定且 fallback 路徑不存在: {_DEFAULT_ROOT}"
        )

KEYWORDS: list[str] = [
    "copernicusmarine",
    "copernicus",
    "marine",
    "cmems",
    "netcdf",
    "keyword_search",
    "false_negative",
]

SCAN_EXTENSIONS: tuple[str, ...] = (".py", ".md", ".txt", ".js", ".html")
SKIP_DIRS: frozenset[str] = frozenset({"venv", ".git", "__pycache__", "node_modules"})

# NetCDF 專用副檔名
NC_EXTENSIONS: tuple[str, ...] = (".nc", ".nc4", ".cdf")


# ---------------------------------------------------------------------------
# 科學驗證輔助函式
# ---------------------------------------------------------------------------
def validate_cpue(cpue: float, species_code: str = "SKJ") -> bool:
    """驗證 CPUE（Catch Per Unit Effort）合理範圍。

    漁業科學意義
    ------------
    CPUE 是漁業資源評估的核心指標。異常值（負值或極端高值）
    通常表示資料輸入錯誤或單位換算錯誤，會嚴重扭曲資源評估模型。

    Parameters
    ----------
    cpue : float
        單位努力漁獲量。
    species_code : str
        FAO 三字母物種代碼（預設 SKJ = 正鰹）。

    Returns
    -------
    bool
        True 表示值在合理範圍內。
    """
    if cpue < 0:
        return False
    # 粗略上界：圍網正鰹單網次 CPUE 通常 < 500 噸
    upper_bounds: dict[str, float] = {"SKJ": 500.0, "YFT": 300.0, "BET": 200.0}
    upper = upper_bounds.get(species_code, 1000.0)
    return cpue <= upper


def validate_nc_quality_flag(flag_value: int) -> bool:
    """驗證 NetCDF 品質旗標是否在 CMEMS 規範的有效值域內。

    漁業科學意義
    ------------
    CMEMS L3/L4 產品的 ``quality_flag`` 遵循 OceanSITES 規範：
      0=unknown, 1=good, 2=probably_good, 3=potentially_correctable,
      4=bad, 5-8=reserved, 9=missing。
    超出此範圍的值代表檔案損壞或版本不相容。

    Parameters
    ----------
    flag_value : int
        NetCDF ``quality_flag`` 整數值。

    Returns
    -------
    bool
    """
    return flag_value in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


def validate_chl_range(chl: float) -> bool:
    """驗證葉綠素-a 濃度範圍（CHL_RANGE 規則）。

    漁業科學意義
    ------------
    葉綠素峰值帶（CHL ≥ 0.2 mg/m³）為圍網與 FAD 漁場核心；
    正鰹漁場常落在 CHL 梯度前緣。超出 0.01–100.0 mg/m³ 的值
    幾乎必定為感測器錯誤或單位錯誤。

    Parameters
    ----------
    chl : float
        葉綠素-a 濃度 (mg/m³)。

    Returns
    -------
    bool
    """
    return 0.01 <= chl <= 100.0


# ---------------------------------------------------------------------------
# NetCDF 錯誤分類
# ---------------------------------------------------------------------------
def classify_nc_error(error: OSError) -> str:
    """將 NetCDF / OS 層級錯誤分類為 timeout / corrupt / version_mismatch。

    漁業科學意義
    ------------
    CMEMS NetCDF 讀取失敗時，須區分三類以利後續重試策略：
    - timeout: 網路中斷可重試
    - corrupt: 檔案損壞須重新下載
    - version_mismatch: 格式版本不符須更新解析器

    Parameters
    ----------
    error : OSError
        捕獲的 OS 層級例外。

    Returns
    -------
    str
        'timeout' | 'corrupt' | 'version_mismatch' | 'unknown_os_error'
    """
    err = getattr(error, "errno", None)
    msg_lower = str(error).lower()

    # 網路逾時
    if err in (errno.ETIMEDOUT, errno.ECONNREFUSED, errno.ECONNRESET):
        return "timeout"

    # HDF / NetCDF-4 版本不符
    if "hdf" in msg_lower or "netcdf-4" in msg_lower or "version" in msg_lower:
        return "version_mismatch"

    # 檔案損壞或截斷
    if "corrupt" in msg_lower or "truncat" in msg_lower or err == errno.EIO:
        return "corrupt"

    # 訊息層級逾時關鍵字（非 errno 驅動）
    if "timeout" in msg_lower or "timed out" in msg_lower:
        return "timeout"

    return "unknown_os_error"