"""
OceanMaster v13.2 — Taiwan Fishery Market Data Integration
=========================================================
抓取台灣農業部漁產品交易行情 API，用實際市場漁獲量
動態校正 CPUE baseline，讓預測與台灣本地實際漁況掛鉤。

API: https://data.moa.gov.tw/Service/OpenData/FromM/AquaticTransData.aspx
格式: JSON，每日更新
欄位: 交易日期、品種代碼、魚貨名稱、市場名稱、上價、中價、下價、交易量、平均價
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path
import json

log = logging.getLogger("OceanMaster.FisheryMarket")

API_URL = "https://data.moa.gov.tw/Service/OpenData/FromM/AquaticTransData.aspx"

# 台灣漁市場魚名 → OceanMaster species key 映射
_NAME_MAP = {
    "黃鰭鮪": "yellowfin",
    "大目鮪": "bigeye",
    "正鰹": "skipjack",
    "鰹": "skipjack",
    "長鰭鮪": "albacore",
    "日本鎖管": "japanese_flying_squid",
    "鎖管": "japanese_flying_squid",
    "秋刀魚": "pacific_saury",
    "黃鰭鮪魚": "yellowfin",
    "大目鮪魚": "bigeye",
    "長鰭鮪魚": "albacore",
}

# 歷史月均漁獲量基準 (kg)，用於計算動態校正係數
# ⚠️ [v16-cal] 來源: 手動估算自台灣遠洋漁業年報 (2018-2022)
#     這些是粗略近似值，非精確統計。優先使用 WCPFC 校準資料。
_HIST_MONTHLY_VOLUME = {
    "yellowfin":  {1: 800, 2: 900, 3: 1100, 4: 1300, 5: 1500, 6: 1400,
                   7: 1200, 8: 1000, 9: 900, 10: 850, 11: 800, 12: 800},
    "bigeye":     {1: 600, 2: 650, 3: 700, 4: 800, 5: 900, 6: 850,
                   7: 750, 8: 700, 9: 650, 10: 600, 11: 580, 12: 600},
    "skipjack":   {1: 1500, 2: 1800, 3: 2000, 4: 2200, 5: 2500, 6: 2300,
                   7: 2000, 8: 1800, 9: 1500, 10: 1400, 11: 1300, 12: 1400},
    "albacore":   {1: 400, 2: 450, 3: 500, 4: 600, 5: 700, 6: 650,
                   7: 550, 8: 500, 9: 450, 10: 400, 11: 380, 12: 400},
}
_HIST_MONTHLY_VOLUME_SOURCE = "manual_estimate_2018-2022"


def fetch_fishery_market_data(lookback_days: int = 90) -> Dict:
    """
    Fetch recent fishery market transaction data from Taiwan MOA API.

    Returns:
        {
            "yellowfin": {"volume_kg": 2500, "avg_price": 320, "months_data": {...}},
            "bigeye": {...},
            ...
            "_meta": {"source": "台灣農業部", "records": 150, "date_range": "..."}
        }
    """
    try:
        import httpx
    except ImportError:
        import requests as httpx

    cache_path = Path("data/cache/fishery_market.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Check cache (24h)
    if cache_path.exists():
        age_h = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_h < 24:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                log.info(f"  🐟 漁市行情: cache hit ({age_h:.1f}h old, {cached.get('_meta', {}).get('records', '?')} records)")
                return cached
            except Exception as e:
                log.debug(f"[降級] engine/fishery_market.py: {e}")

    result = {}
    try:
        log.info(f"  🐟 正在抓取台灣漁產品交易行情 ({API_URL[:50]}...)")
        resp = httpx.get(API_URL, timeout=30)
        if hasattr(resp, 'raise_for_status'):
            resp.raise_for_status()

        data = resp.json() if hasattr(resp, 'json') else json.loads(resp.text)

        if not isinstance(data, list) or len(data) == 0:
            log.warning("  漁市行情: API 回傳空數據")
            return _fallback_market_data()

        # Parse records
        species_agg = {}  # {species: {month: [volumes]}}
        species_prices = {}  # {species: {month: [prices]}}
        n_matched = 0

        for rec in data:
            name = rec.get("魚貨名稱", "")
            sp = None
            for fish_name, sp_key in _NAME_MAP.items():
                if fish_name in name:
                    sp = sp_key
                    break
            if sp is None:
                continue

            try:
                vol = float(rec.get("交易量", 0))
                price = float(rec.get("平均價", 0))
                date_str = rec.get("交易日期", "")
                if date_str:
                    # Parse date to get month
                    # Format varies: could be YYYY/MM/DD or YYYY-MM-DD or YYYYMMDD
                    date_str_clean = date_str.replace("/", "-").replace(".", "-")
                    try:
                        dt = datetime.strptime(date_str_clean[:10], "%Y-%m-%d")
                    except ValueError:
                        try:
                            # ROC calendar (民國年): 113/01/15 → 2024/01/15
                            parts = date_str.split("/")
                            if len(parts) == 3 and int(parts[0]) < 200:
                                yr = int(parts[0]) + 1911
                                dt = datetime(yr, int(parts[1]), int(parts[2]))
                            else:
                                continue
                        except Exception:
                            continue

                    mon = dt.month
                    if sp not in species_agg:
                        species_agg[sp] = {}
                        species_prices[sp] = {}
                    species_agg[sp].setdefault(mon, []).append(vol)
                    species_prices[sp].setdefault(mon, []).append(price)
                    n_matched += 1
            except (ValueError, TypeError):
                continue

        # Aggregate
        for sp, months in species_agg.items():
            total_vol = sum(sum(vols) for vols in months.values())
            all_prices = [p for plist in species_prices.get(sp, {}).values() for p in plist if p > 0]
            avg_price = sum(all_prices) / len(all_prices) if all_prices else 0

            result[sp] = {
                "volume_kg": round(total_vol, 0),
                "avg_price": round(avg_price, 1),
                "months_data": {m: round(sum(vols), 0) for m, vols in months.items()},
            }

        result["_meta"] = {
            "source": "台灣農業部漁產品交易行情",
            "records": n_matched,
            "total_api_records": len(data),
            "fetched_at": datetime.now().isoformat(),
        }

        log.info(f"  🐟 漁市行情: {n_matched} 筆匹配, {len(result)-1} 魚種, API 共 {len(data)} 筆")

        # Cache
        try:
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            log.debug(f"[降級] engine/fishery_market.py: {e}")

        return result

    except Exception as e:
        log.warning(f"  漁市行情: API 失敗 ({e}), 使用 fallback")
        return _fallback_market_data()


def _fallback_market_data() -> Dict:
    """Fallback with reasonable defaults when API is unavailable."""
    return {
        "_meta": {"source": "fallback (API 離線)", "records": 0},
    }


def calibrate_cpue_baseline(
    baseline: Dict,
    market_data: Dict,
    month: int,
) -> Dict:
    """
    Dynamically adjust CPUE baseline using market volume data.

    If current month's market volume is significantly different from
    historical average, scale the baseline accordingly.

    Args:
        baseline: Original _BASELINE_CPUE dict {species: {month: cpue}}
        market_data: Output from fetch_fishery_market_data()
        month: Current month

    Returns:
        Adjusted baseline dict (same structure)
    """
    if not market_data or market_data.get("_meta", {}).get("records", 0) == 0:
        return baseline, {}

    adjusted = {}
    calibration_log = {}

    for sp, monthly_cpue in baseline.items():
        adjusted[sp] = dict(monthly_cpue)

        market_sp = market_data.get(sp)
        if market_sp is None:
            continue

        months_data = market_sp.get("months_data", {})
        hist = _HIST_MONTHLY_VOLUME.get(sp, {})

        # Get current month's market volume
        current_vol = months_data.get(month, 0)
        hist_vol = hist.get(month, 1)

        if current_vol > 0 and hist_vol > 0:
            # ratio = current / historical
            # ratio > 1 = more fish being caught = higher CPUE
            # ratio < 1 = less fish = lower CPUE
            ratio = current_vol / hist_vol
            # Dampen: sqrt(ratio) to avoid wild swings
            factor = ratio ** 0.5
            # Clamp to [0.5, 2.0]
            factor = max(0.5, min(2.0, factor))

            adjusted[sp][month] = round(monthly_cpue.get(month, 2.0) * factor, 2)
            calibration_log[sp] = {
                "market_vol": current_vol,
                "hist_vol": hist_vol,
                "ratio": round(ratio, 2),
                "factor": round(factor, 2),
                "original": monthly_cpue.get(month, 2.0),
                "adjusted": adjusted[sp][month],
            }

    if calibration_log:
        for sp, cal in calibration_log.items():
            log.info(f"  📊 {sp}: 行情校正 {cal['original']} → {cal['adjusted']} "
                     f"(市場量 {cal['market_vol']:.0f}kg / 歷史 {cal['hist_vol']}kg = "
                     f"×{cal['factor']:.2f})")

    return adjusted, calibration_log
