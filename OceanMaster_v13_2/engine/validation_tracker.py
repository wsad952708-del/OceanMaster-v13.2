"""
OceanMaster v13.2 — 預測驗證追蹤器
==================================
自動追蹤預測準確度，計算 Brier Score / Spearman ρ / Hit Rate。

商業用途:
  - 持續驗證預測品質 → 建立信譽
  - 自動發現模型退化 → 觸發再訓練
  - 提供給客戶的準確率報告
"""

import numpy as np
import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("OceanMaster.Validation")


class PredictionValidator:
    """
    預測驗證追蹤器

    使用:
        validator = PredictionValidator()
        validator.record_prediction(lat, lon, date, species, predicted_score, predicted_hotspot)
        validator.record_observation(lat, lon, date, species, actual_cpue)
        metrics = validator.compute_metrics()
    """

    RECORDS_FILE = "data/validation_records.jsonl"

    def __init__(self, records_file: str = None):
        self.records_file = records_file or self.RECORDS_FILE
        Path(self.records_file).parent.mkdir(parents=True, exist_ok=True)
        self.predictions = []
        self.observations = []

    def record_prediction(
        self,
        lat: float, lon: float,
        date: str,
        species: str,
        predicted_score: float,
        is_hotspot: bool,
        confidence_lower: float = None,
        confidence_upper: float = None,
    ):
        """記錄一筆預測"""
        record = {
            "type": "prediction",
            "lat": lat, "lon": lon,
            "date": date,
            "species": species,
            "score": predicted_score,
            "hotspot": is_hotspot,
            "ci_lower": confidence_lower,
            "ci_upper": confidence_upper,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.predictions.append(record)
        self._append_to_file(record)

    def record_observation(
        self,
        lat: float, lon: float,
        date: str,
        species: str,
        actual_cpue: float,
        catch_kg: float = 0,
        effort_hours: float = 0,
    ):
        """記錄一筆實際觀測/漁獲"""
        record = {
            "type": "observation",
            "lat": lat, "lon": lon,
            "date": date,
            "species": species,
            "cpue": actual_cpue,
            "catch_kg": catch_kg,
            "effort_hours": effort_hours,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.observations.append(record)
        self._append_to_file(record)

    def compute_metrics(self, species: str = None) -> Dict:
        """
        計算驗證指標

        Returns:
            dict with:
              - brier_score: 0-1 (越低越好，0=完美)
              - spearman_rho: -1~1 (越高越好)
              - hit_rate: 0-1 (預測熱點實際有漁獲的比例)
              - n_matched: 配對成功的記錄數
              - ci_coverage: 信賴區間覆蓋率 (應 ~90%)
        """
        matched = self._match_predictions_observations(species)
        if len(matched) < 5:
            return {
                "brier_score": None,
                "spearman_rho": None,
                "hit_rate": None,
                "n_matched": len(matched),
                "ci_coverage": None,
                "status": "insufficient_data",
            }

        pred_scores = np.array([m["pred_score"] for m in matched])
        pred_hotspots = np.array([m["pred_hotspot"] for m in matched])
        actual_cpues = np.array([m["obs_cpue"] for m in matched])

        # 二值化觀測 (CPUE > 中位數 = 好漁場)
        cpue_median = np.median(actual_cpues)
        actual_good = (actual_cpues > cpue_median).astype(float)

        # Brier Score
        brier = np.mean((pred_scores - actual_good) ** 2)

        # Spearman rank correlation
        try:
            from scipy.stats import spearmanr
            rho, p_value = spearmanr(pred_scores, actual_cpues)
        except ImportError:
            # 手動 Spearman
            n = len(pred_scores)
            rank_pred = np.argsort(np.argsort(pred_scores)) + 1
            rank_obs = np.argsort(np.argsort(actual_cpues)) + 1
            d_sq = np.sum((rank_pred - rank_obs) ** 2)
            rho = 1 - 6 * d_sq / (n * (n**2 - 1))
            p_value = None

        # Hit Rate (預測熱點中實際 CPUE > 中位數的比例)
        hotspot_mask = pred_hotspots.astype(bool)
        if np.any(hotspot_mask):
            hit_rate = np.mean(actual_cpues[hotspot_mask] > cpue_median)
        else:
            hit_rate = None

        # 信賴區間覆蓋率
        ci_count = 0
        ci_total = 0
        for m in matched:
            if m.get("ci_lower") is not None and m.get("ci_upper") is not None:
                ci_total += 1
                if m["ci_lower"] <= m["obs_cpue"] <= m["ci_upper"]:
                    ci_count += 1
        ci_coverage = ci_count / ci_total if ci_total > 0 else None

        metrics = {
            "brier_score": float(brier),
            "spearman_rho": float(rho) if rho is not None else None,
            "hit_rate": float(hit_rate) if hit_rate is not None else None,
            "n_matched": len(matched),
            "ci_coverage": ci_coverage,
            "status": "ok",
        }

        log.info(f"  Validation metrics: Brier={brier:.3f}, ρ={rho:.3f}, "
                 f"Hit={hit_rate:.1%}, n={len(matched)}")

        return metrics

    def _match_predictions_observations(
        self, species: str = None, max_dist_km: float = 50, max_days: int = 2
    ) -> List[Dict]:
        """配對預測和觀測記錄 (空間+時間鄰近)"""
        matched = []
        for obs in self.observations:
            if species and obs["species"] != species:
                continue
            best_pred = None
            best_dist = float("inf")
            for pred in self.predictions:
                if pred["species"] != obs["species"]:
                    continue
                if pred["date"] != obs["date"]:
                    continue
                dist = self._haversine(
                    obs["lat"], obs["lon"], pred["lat"], pred["lon"]
                )
                if dist < best_dist and dist < max_dist_km:
                    best_dist = dist
                    best_pred = pred
            if best_pred is not None:
                matched.append({
                    "pred_score": best_pred["score"],
                    "pred_hotspot": best_pred["hotspot"],
                    "obs_cpue": obs["cpue"],
                    "ci_lower": best_pred.get("ci_lower"),
                    "ci_upper": best_pred.get("ci_upper"),
                    "dist_km": best_dist,
                })
        return matched

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """兩點間距離 (km)"""
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def _append_to_file(self, record):
        """追加記錄到 JSONL 檔案"""
        try:
            with open(self.records_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"驗證記錄寫入失敗: {e}")
