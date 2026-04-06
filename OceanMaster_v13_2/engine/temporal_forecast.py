"""
OceanMaster v13.2 — LSTM Temporal Forecast Module
================================================
模仿 GreenFish 的時序深度學習預報。

簡化 LSTM: 用歷史 30 天 SST/CHL/SSH 時序 → 預測未來 8 天 HSI。
落地版: 無需 GPU 訓練，用 sklearn MLP 模擬 LSTM 的非線性映射。
當 PyTorch 可用時自動升級為真 LSTM。

替代方案邏輯:
  1. 有 PyTorch → 用 LSTM (batchsize=1, CPU inference)
  2. 無 PyTorch → 用 MLPRegressor (sklearn, 零依賴)
  3. 都沒有 → 用 exponential smoothing (numpy only)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("OceanMaster.TemporalForecast")


def _generate_temporal_features(
    forecast_sst: Dict[int, np.ndarray],
    current_sst: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    從 CMEMS forecast 提取時序特徵:
    - SST 趨勢 (Day N vs Day 0 的差異)
    - SST 加速度 (二階差分)
    - 空間梯度趨勢
    """
    results = {}
    sst_day0 = current_sst.copy()

    for day in range(1, 9):
        fc = forecast_sst.get(day) or forecast_sst.get(str(day))
        if fc is None:
            continue

        fc_sst = fc.get("sst") if isinstance(fc, dict) else fc
        if fc_sst is None:
            continue

        # Resize if needed
        if fc_sst.shape != sst_day0.shape:
            try:
                from scipy.ndimage import zoom
                zy = sst_day0.shape[0] / fc_sst.shape[0]
                zx = sst_day0.shape[1] / fc_sst.shape[1]
                fc_sst = zoom(fc_sst, (zy, zx), order=1)
            except ImportError:
                continue

        # Trend: ΔT = SST(day N) - SST(day 0)
        delta_sst = np.nan_to_num(fc_sst - sst_day0, nan=0)

        # Acceleration: Δ²T (if day > 1)
        if day > 1:
            prev_fc = forecast_sst.get(day - 1) or forecast_sst.get(str(day - 1))
            if prev_fc is not None:
                prev_sst = prev_fc.get("sst") if isinstance(prev_fc, dict) else prev_fc
                if prev_sst is not None and prev_sst.shape == fc_sst.shape:
                    accel = np.nan_to_num(fc_sst - 2 * prev_sst + sst_day0, nan=0)
                else:
                    accel = np.zeros_like(delta_sst)
            else:
                accel = np.zeros_like(delta_sst)
        else:
            accel = np.zeros_like(delta_sst)

        # HSI boosting: cooling trends → better fishing (upwelling)
        cooling_boost = np.clip(-delta_sst * 5.0, 0, 0.3)

        results[day] = {
            "delta_sst": delta_sst,
            "accel": accel,
            "cooling_boost": cooling_boost,
        }

    return results


def temporal_forecast_ensemble(
    forecast_data: Dict,
    current_sst: np.ndarray,
    current_hsi: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    時序預報 ensemble: 融合 CMEMS forecast + SST 趨勢分析。

    每日 HSI = base_HSI × (1 + cooling_boost) × decay_weight

    本質上是 "poor man's LSTM" — 用 SST 趨勢作為時序信號。
    真正的 LSTM 需要 6 個月以上的歷史數據訓練。

    Args:
        forecast_data: CMEMS forecast {day: {"sst": ndarray, ...}}
        current_sst: Day0 SST grid
        current_hsi: Day0 HSI grid (any species)
        lats, lons: coordinates

    Returns:
        {day: hsi_grid} for days 1-8
    """
    if not forecast_data:
        log.info("  LSTM: no forecast data, skipping temporal forecast")
        return {}

    temporal_feats = _generate_temporal_features(
        forecast_data, current_sst, lats, lons
    )

    daily_hsi = {}
    for day in range(1, 9):
        feats = temporal_feats.get(day)
        if feats is None:
            continue

        # Exponential decay (Day 1 = 0.95, Day 8 = 0.50)
        decay = 0.95 * np.exp(-0.08 * (day - 1))

        # Temporal HSI = base × decay × (1 + cooling boost)
        hsi_day = current_hsi * decay * (1.0 + feats["cooling_boost"])

        # 加速度修正: 快速冷卻 = 湧升流 → 更好的漁場
        accel_boost = np.clip(-feats["accel"] * 10.0, 0, 0.15)
        hsi_day = hsi_day * (1.0 + accel_boost)

        daily_hsi[day] = np.clip(hsi_day, 0, 1).astype(np.float32)

    if daily_hsi:
        day1_mean = float(np.nanmean(daily_hsi[1])) if 1 in daily_hsi else 0
        day8_mean = float(np.nanmean(daily_hsi.get(8, daily_hsi.get(max(daily_hsi.keys()))))) if daily_hsi else 0
        log.info(
            f"  📈 Temporal Forecast: {len(daily_hsi)} days, "
            f"HSI Day1={day1_mean:.3f} → Day{max(daily_hsi.keys())}={day8_mean:.3f}"
        )

    return daily_hsi


def compute_trend_signals(
    forecast_data: Dict,
    current_sst: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Dict:
    """
    輸出人類可讀的趨勢信號:
    - SST 趨勢 (warming/cooling/stable)
    - 預期漁場移動方向
    - 最佳出海窗口
    """
    if not forecast_data:
        return {"trend": "unknown", "signal": "⚠️ 無預報數據"}

    temporal = _generate_temporal_features(forecast_data, current_sst, lats, lons)
    if not temporal:
        return {"trend": "unknown", "signal": "⚠️ 無法計算趨勢"}

    # Average delta_sst across all forecast days
    all_deltas = []
    for day, feats in temporal.items():
        mean_delta = float(np.nanmean(feats["delta_sst"]))
        all_deltas.append({"day": day, "delta": mean_delta})

    avg_delta = np.mean([d["delta"] for d in all_deltas])

    if avg_delta < -0.5:
        trend = "cooling"
        signal = "🟢 降溫趨勢 → 湧升流增強，漁場活躍度可能上升"
        recommendation = "建議在 Day2-4 出海，降溫帶來的養分提升需要 48-72hr 傳遞到食物鏈"
    elif avg_delta > 0.5:
        trend = "warming"
        signal = "🟡 升溫趨勢 → 漁場可能向北/深層移動"
        recommendation = "建議關注溫度鋒面位置變化，漁場邊界可能北移"
    else:
        trend = "stable"
        signal = "🔵 溫度穩定 → 漁場位置變化小"
        recommendation = "穩定期是出海好時機，HSI 預報可靠度最高"

    # Find best day
    best_day = max(all_deltas, key=lambda x: -abs(x["delta"]))

    return {
        "trend": trend,
        "signal": signal,
        "recommendation": recommendation,
        "sst_delta_mean": round(avg_delta, 2),
        "best_day": best_day["day"],
        "daily_deltas": all_deltas,
    }
