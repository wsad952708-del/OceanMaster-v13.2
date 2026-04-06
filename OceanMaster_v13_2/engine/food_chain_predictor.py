"""
OceanMaster v13.2 — 食物鏈級聯預測引擎 (Food Chain Cascade Predictor)  # [v12-phase8]
=================================================================================
核心創新: 營養級級聯時序模型 (Trophic Cascade Timing Model)

  日照 + 營養鹽 → 浮游植物爆發 (0 天)
    → 浮游動物高峰 (7-14 天延遲, Platt et al. 2003)
      → 微型浮游生物/餌料魚聚集 (14-21 天延遲)
        → 鮪魚抵達覓食 (21-35 天延遲)

科學參考:
  - 浮游植物-浮游動物延遲: Henson et al. 2009, JGR-Oceans
  - 爆發偵測啟動: Brody et al. 2013, JGR-Oceans
  - 營養級級聯: Platt et al. 2003, Nature
  - 鮪魚聚集時序: Precioso et al. 2022 (TUN-AI), Fisheries Research
  - 葉綠素分級: Brewin et al. 2010, JGR-Oceans
  - OMZ 增益: Stramma et al. 2012, Nature Climate Change
"""

import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import config  # [v12-phase8] 引入全域配置

log = logging.getLogger("OceanMaster.FoodChain")


# [v12-phase8] 物種特異性時序參數
SPECIES_TROPHIC_TIMING = {
    "yellowfin": {
        "name_zh": "黃鰭鮪",
        "zoo_lag_warm": (5, 10),    # 暖水 (>25°C) 浮游動物延遲 (天)
        "zoo_lag_cool": (10, 18),   # 溫水 (15-25°C)
        "zoo_lag_cold": (18, 30),   # 冷水 (<15°C)
        "arrival_after_zoo": (3, 7),  # 浮游動物高峰後鮪魚到達 (天)
        "feeding_style": "aggressive",
        "depth_range": (0, 400),
        "phi_threshold": 3.0,
        "speed_km_day": (30, 50),
    },
    "bigeye": {
        "name_zh": "大目鮪",
        "zoo_lag_warm": (5, 10),
        "zoo_lag_cool": (10, 18),
        "zoo_lag_cold": (18, 30),
        "arrival_after_zoo": (5, 10),
        "feeding_style": "deep_ambush",
        "depth_range": (100, 500),
        "phi_threshold": 2.5,
        "speed_km_day": (20, 40),
    },
    "skipjack": {
        "name_zh": "正鰹",
        "zoo_lag_warm": (5, 10),
        "zoo_lag_cool": (10, 18),
        "zoo_lag_cold": (18, 30),
        "arrival_after_zoo": (1, 5),
        "feeding_style": "surface_fast",
        "depth_range": (0, 200),
        "phi_threshold": 3.5,
        "speed_km_day": (40, 60),
    },
    "albacore": {
        "name_zh": "長鰭鮪",
        "zoo_lag_warm": (5, 10),
        "zoo_lag_cool": (10, 18),
        "zoo_lag_cold": (18, 30),
        "arrival_after_zoo": (7, 14),
        "feeding_style": "thermal_front_follower",
        "depth_range": (50, 300),
        "phi_threshold": 2.8,
        "speed_km_day": (50, 80),
    },
}


class FoodChainPredictor:
    """
    [v12-phase8] 食物鏈級聯預測引擎

    提供完整的營養級時序預測:
    1. 浮游植物爆發階段估計
    2. 浮游動物響應預測
    3. 鮪魚到達時間估計
    4. 完整食物鏈時間線

    這是 OceanMaster 的核心差異化功能。
    沒有其他系統提供完整的營養級級聯時間線。
    """

    # ─── 爆發判定閾值 (Brody et al. 2013) ───
    BLOOM_INITIATION_RATIO = 1.5   # Chl > 1.5× 氣候態 = 啟動
    BLOOM_PEAK_RATIO = 3.0         # Chl > 3× 氣候態 = 高峰
    BLOOM_DECLINE_RATE = -0.05     # 日變化率 < -5% = 衰退

    def estimate_bloom_stage(
        self,
        chl_current: float,
        chl_history_7d: List[float],
        chl_climatology: float,
    ) -> Dict:
        """
        [v12-phase8] 估計浮游植物爆發階段

        Reference: Brody et al. 2013 "A comparison of methods for
                   determining phytoplankton bloom initiation"

        Parameters:
            chl_current: 當前 Chl-a (mg/m³)
            chl_history_7d: 過去 7 天的 Chl-a 歷史 (index 0 = 7 天前, -1 = 昨天)
            chl_climatology: 氣候態 Chl-a (mg/m³)

        Returns:
            dict with bloom_phase, bloom_intensity, days_since_initiation,
            estimated_peak_date, trend
        """
        # 安全處理
        chl_current = max(chl_current, 0.001)
        chl_climatology = max(chl_climatology, 0.01)
        if not chl_history_7d:
            chl_history_7d = [chl_current]

        ratio = chl_current / chl_climatology
        now = datetime.now(timezone.utc)

        # 計算趨勢 (線性回歸斜率)
        if len(chl_history_7d) >= 2:
            x = np.arange(len(chl_history_7d))
            y = np.array(chl_history_7d, dtype=float)
            y = np.where(y > 0, y, 0.001)  # 數值安全
            # 用 log 空間計算趨勢 (Chl 呈對數正態)
            log_y = np.log10(y)
            slope = float(np.polyfit(x, log_y, 1)[0])
            # 轉換為每日變化率
            daily_rate = slope / max(len(chl_history_7d), 1)
        else:
            daily_rate = 0.0
            slope = 0.0

        is_rising = daily_rate > 0.01
        is_plateauing = abs(daily_rate) < 0.01
        is_declining = daily_rate < self.BLOOM_DECLINE_RATE

        # 找爆發啟動點 (ratio 首次超過 1.5)
        days_since_initiation = 0
        for i, val in enumerate(chl_history_7d):
            if val / chl_climatology >= self.BLOOM_INITIATION_RATIO:
                days_since_initiation = len(chl_history_7d) - i
                break

        # 判定階段
        if ratio < 1.2:
            if is_rising and ratio > 0.9:
                phase = "pre-bloom"
                estimated_peak_days = int(7 + (self.BLOOM_PEAK_RATIO - ratio) / max(daily_rate * 10, 0.01))
            else:
                phase = "post-bloom" if is_declining else "pre-bloom"
                estimated_peak_days = 14
        elif ratio < self.BLOOM_PEAK_RATIO:
            if is_rising:
                phase = "initiation"
                # 估計到達峰值的天數
                remaining_ratio = self.BLOOM_PEAK_RATIO - ratio
                if daily_rate > 0:
                    divisor = max(daily_rate * 10 * chl_climatology / chl_current, 0.001)  # [v12-phase8-xval] 防零除
                    estimated_peak_days = int(remaining_ratio / divisor)
                    estimated_peak_days = np.clip(estimated_peak_days, 1, 21)
                else:
                    estimated_peak_days = 7
            elif is_declining:
                phase = "decline"
                estimated_peak_days = -days_since_initiation  # 已過峰值
            else:
                phase = "initiation"
                estimated_peak_days = 5
        else:  # ratio >= 3.0
            if is_plateauing or is_rising:
                phase = "peak"
                estimated_peak_days = 0
            else:
                phase = "decline"
                estimated_peak_days = -2  # 剛過峰值

        # 爆發強度 (0-1)
        bloom_intensity = float(np.clip((ratio - 1.0) / 4.0, 0.0, 1.0))

        estimated_peak_date = now + timedelta(days=max(estimated_peak_days, 0))

        return {
            "bloom_phase": phase,
            "bloom_intensity": round(bloom_intensity, 3),
            "days_since_initiation": max(days_since_initiation, 0),
            "estimated_peak_date": estimated_peak_date.strftime("%Y-%m-%d"),
            "chl_ratio": round(ratio, 2),
            "daily_trend": round(daily_rate, 4),
            "trend_direction": "rising" if is_rising else ("declining" if is_declining else "stable"),
        }

    def estimate_zooplankton_response(
        self,
        bloom_phase: str,
        sst: float,
        days_since_bloom: int,
    ) -> Dict:
        """
        [v12-phase8] 估計浮游動物響應

        浮游動物對浮游植物的響應具有溫度依賴性的延遲。
        Reference: Henson et al. 2009 "Timing of
                   phytoplankton-zooplankton coupling"

        Parameters:
            bloom_phase: 爆發階段 (from estimate_bloom_stage)
            sst: 海表溫度 (°C)
            days_since_bloom: 自爆發啟動以來的天數

        Returns:
            dict with zoo_biomass_index, zoo_peak_eta_days,
            dominant_size_class
        """
        # [v12-phase8-xval] 由 config.FOOD_CHAIN_TIMING 提取參數 (預設 yellowfin 作為基準)
        fc_params = config.FOOD_CHAIN_TIMING.get("yellowfin", {})

        # 溫度依賴的延遲估計 (Henson 2009)
        if sst > 25.0:
            lag_range = fc_params.get("zoo_lag_warm", (5, 10))
            t_label = "warm"
        elif sst > 15.0:
            lag_range = fc_params.get("zoo_lag_cool", (10, 18))
            t_label = "cool"
        else:
            lag_range = fc_params.get("zoo_lag_cold", (18, 30))
            t_label = "cold"

        lag_mid = (lag_range[0] + lag_range[1]) / 2.0
        lag_sigma = (lag_range[1] - lag_range[0]) / 4.0  # ~95% 在範圍內

        # 浮游動物生物量指數 (高斯響應曲線)
        if bloom_phase in ("initiation", "peak", "decline"):
            # 以延遲中值為中心的高斯
            zoo_response = np.exp(-0.5 * ((days_since_bloom - lag_mid) / max(lag_sigma, 1)) ** 2)
            zoo_biomass_index = float(np.clip(zoo_response, 0.0, 1.0))
        elif bloom_phase == "pre-bloom":
            zoo_biomass_index = 0.1  # 背景水平
        else:  # post-bloom
            # 衰減但仍有殘留
            decay = max(0.0, 1.0 - (days_since_bloom - lag_mid * 1.5) / lag_mid)
            zoo_biomass_index = float(np.clip(decay * 0.6, 0.05, 0.6))

        # 預計浮游動物高峰 ETA
        zoo_peak_eta = max(0, int(lag_mid - days_since_bloom))

        # 主導體型 (與 SST 相關)
        if sst > 26.0:
            size_class = "micro"   # 暖水=小型浮游動物
        elif sst > 18.0:
            size_class = "meso"    # 溫水=中型
        else:
            size_class = "macro"   # 冷水=大型 (磷蝦等)

        return {
            "zoo_biomass_index": round(zoo_biomass_index, 3),
            "zoo_peak_eta_days": zoo_peak_eta,
            "dominant_size_class": size_class,
            "temperature_regime": t_label,
            "lag_range_days": list(lag_range),
            "zoo_status": "building" if days_since_bloom < lag_mid else (
                "peak" if abs(days_since_bloom - lag_mid) < lag_sigma else "declining"
            ),
        }

    def estimate_tuna_arrival(
        self,
        zoo_biomass_index: float,
        phi: float,
        front_strength: float,
        eddy_strength: float,
        dvm_depth: float,
        species: str,
        zoo_trend: str = "stable",
    ) -> Dict:
        """
        [v12-phase8] 估計鮪魚到達時間

        鮪魚跟隨獵物濃度。較高機率的條件:
        - 浮游動物生物量 > 0.6 且上升中
        - 附近有活躍鋒面（獵物聚集）
        - Φ > 3.0（代謝適宜）
        - 適度渦旋活動（滯留效應）
        - DVM 深度在物種可達範圍

        Reference: Precioso et al. 2022 (TUN-AI), Fisheries Research

        Parameters:
            zoo_biomass_index: 浮游動物生物量指數 (0-1)
            phi: 代謝指數 Φ
            front_strength: 鋒面強度 (0-1)
            eddy_strength: 渦旋強度 (0-1)
            dvm_depth: DVM 深度 (m)
            species: 物種 key
            zoo_trend: "building" | "peak" | "declining" | "stable"

        Returns:
            dict with feeding_probability, estimated_arrival_window,
            optimal_fishing_date, confidence
        """
        # [v12-phase8-xval] 改用 config 中的 FOOD_CHAIN_TIMING
        fc_timing = config.FOOD_CHAIN_TIMING.get(species, config.FOOD_CHAIN_TIMING["yellowfin"])
        sp = SPECIES_TROPHIC_TIMING.get(species, SPECIES_TROPHIC_TIMING["yellowfin"])
        now = datetime.now(timezone.utc)

        # ── 覓食機率計算 ──
        # 基礎分數: 浮游動物生物量
        p_zoo = zoo_biomass_index

        # 浮游動物趨勢加成
        trend_bonus = {"building": 0.15, "peak": 0.10, "declining": -0.05, "stable": 0.0}
        p_zoo += trend_bonus.get(zoo_trend, 0.0)

        # 鋒面加成 (獵物聚集)
        p_front = front_strength * 0.3

        # 代謝適宜性
        phi_threshold = sp["phi_threshold"]
        p_phi = float(np.clip((phi - phi_threshold * 0.5) / phi_threshold, 0.0, 0.3))

        # 渦旋滯留效應
        p_eddy = eddy_strength * 0.15

        # DVM 可及性 (物種深度範圍)
        depth_min, depth_max = sp["depth_range"]
        if depth_min <= dvm_depth <= depth_max:
            p_dvm = 0.1
        else:
            p_dvm = -0.1

        feeding_probability = float(np.clip(
            p_zoo + p_front + p_phi + p_eddy + p_dvm, 0.0, 1.0
        ))

        # ── 到達時間窗 ──
        arr_min = fc_timing.get("arrival_min", 3)
        arr_max = fc_timing.get("arrival_max", 7)

        # 根據當前條件調整
        if zoo_trend == "peak":
            # 浮游動物已達高峰 → 鮪魚可能快到了
            arrival_window = (max(1, arr_min - 2), arr_max - 1)
        elif zoo_trend == "building":
            # 還在建立 → 要等更久
            arrival_window = (arr_min, arr_max + 3)
        else:
            arrival_window = (arr_min, arr_max)

        optimal_days = int((arrival_window[0] + arrival_window[1]) / 2)
        optimal_date = now + timedelta(days=optimal_days)

        # ── 信心度 ──
        # 高信心 = 多重正面信號同時出現
        positive_signals = sum([
            zoo_biomass_index > 0.5,
            front_strength > 0.3,
            phi > phi_threshold,
            eddy_strength > 0.2,
            depth_min <= dvm_depth <= depth_max,
        ])
        confidence = float(np.clip(positive_signals / 5.0 + 0.2, 0.2, 0.95))

        return {
            "feeding_probability": round(feeding_probability, 3),
            "estimated_arrival_window": [arr_min, arr_max],  # [v12-phase8-xval] Fix type issue
            "optimal_fishing_date": optimal_date.strftime("%Y-%m-%d"),
            "confidence": round(confidence, 3),
            "species": species,
            "species_zh": sp["name_zh"],
            "feeding_style": sp["feeding_style"],
            "positive_signals": positive_signals,
        }

    def compute_food_chain_timeline(
        self,
        lat: float,
        lon: float,
        date: Optional[datetime],
        species: str,
        ocean_data: Optional[Dict] = None,
    ) -> Dict:
        """
        [v12-phase8] 計算完整食物鏈時間線

        整合所有子模型，產出完整的營養級級聯時間線。

        Parameters:
            lat, lon: 位置
            date: 日期 (None = now)
            species: 物種 key
            ocean_data: 海洋環境數據 dict (可選; keys: sst, chl, chl_7d,
                        chl_clim, phi, front_strength, eddy_strength, dvm_depth)

        Returns:
            完整時間線 dict
        """
        if date is None:
            date = datetime.now(timezone.utc)

        sp = SPECIES_TROPHIC_TIMING.get(species, SPECIES_TROPHIC_TIMING["yellowfin"])

        # 提取或使用預設海洋數據
        if ocean_data is None:
            ocean_data = {}

        sst = ocean_data.get("sst", 27.0)
        chl = ocean_data.get("chl", 0.3)
        chl_7d = ocean_data.get("chl_7d", [0.2, 0.22, 0.25, 0.28, 0.30, 0.28, 0.30])
        chl_clim = ocean_data.get("chl_clim", 0.2)
        phi = ocean_data.get("phi", 4.0)
        front_strength = ocean_data.get("front_strength", 0.3)
        eddy_strength = ocean_data.get("eddy_strength", 0.2)
        dvm_depth = ocean_data.get("dvm_depth", 100.0)

        # 步驟 1: 爆發階段
        bloom = self.estimate_bloom_stage(chl, chl_7d, chl_clim)

        # 步驟 2: 浮游動物響應
        zoo = self.estimate_zooplankton_response(
            bloom["bloom_phase"], sst, bloom["days_since_initiation"]
        )

        # 步驟 3: 鮪魚到達
        tuna = self.estimate_tuna_arrival(
            zoo["zoo_biomass_index"], phi, front_strength,
            eddy_strength, dvm_depth, species, zoo["zoo_status"]
        )

        # 判定當前階段
        if bloom["bloom_phase"] == "pre-bloom":
            current_stage = "nutrient_accumulation"
            stage_zh = "營養鹽累積期"
        elif bloom["bloom_phase"] in ("initiation", "peak"):
            if zoo["zoo_biomass_index"] < 0.3:
                current_stage = "phytoplankton_bloom"
                stage_zh = "浮游植物爆發"
            else:
                current_stage = "zooplankton_building"
                stage_zh = "浮游動物建立"
        elif bloom["bloom_phase"] == "decline":
            if zoo["zoo_status"] == "peak":
                current_stage = "zooplankton_peak"
                stage_zh = "浮游動物高峰"
            elif tuna["feeding_probability"] > 0.5:
                current_stage = "tuna_feeding"
                stage_zh = "鮪魚覓食中"
            else:
                current_stage = "baitfish_aggregation"
                stage_zh = "餌料魚聚集"
        else:
            current_stage = "post_bloom_recovery"
            stage_zh = "爆發後恢復"

        # 生成建議
        if tuna["feeding_probability"] > 0.7:
            recommendation = (
                f"立即出發！{sp['name_zh']}覓食機率高 ({tuna['feeding_probability']*100:.0f}%)。"
                f"最佳日期: {tuna['optimal_fishing_date']}"
            )
        elif tuna["feeding_probability"] > 0.4:
            arr = tuna["estimated_arrival_window"]
            recommendation = (
                f"預先部署 ({arr[0]}-{arr[1]} 天內)。"
                f"浮游動物{zoo['zoo_status']}中，"
                f"{sp['name_zh']}預計 {tuna['optimal_fishing_date']} 到達。"
            )
        elif bloom["bloom_phase"] in ("initiation", "peak"):
            recommendation = (
                f"浮游植物爆發{bloom['bloom_phase']}階段。"
                f"浮游動物尚未響應，需等待 {zoo['zoo_peak_eta_days']} 天。"
                f"建議持續監控。"
            )
        else:
            recommendation = (
                f"目前食物鏈活動不活躍。"
                f"等待下次浮游植物爆發啟動。"
            )

        # 食物鏈綜合分數
        food_chain_score = float(np.clip(
            bloom["bloom_intensity"] * 0.2
            + zoo["zoo_biomass_index"] * 0.3
            + tuna["feeding_probability"] * 0.5,
            0.0, 1.0
        ))

        return {
            "lat": lat,
            "lon": lon,
            "date": date.strftime("%Y-%m-%d"),
            "species": species,
            "species_zh": sp["name_zh"],
            "current_stage": current_stage,
            "current_stage_zh": stage_zh,
            "bloom": {
                "phase": bloom["bloom_phase"],
                "intensity": bloom["bloom_intensity"],
                "peak_date": bloom["estimated_peak_date"],
                "chl_ratio": bloom["chl_ratio"],
                "trend": bloom["trend_direction"],
            },
            "zooplankton": {
                "biomass_index": zoo["zoo_biomass_index"],
                "peak_eta_days": zoo["zoo_peak_eta_days"],
                "size_class": zoo["dominant_size_class"],
                "status": zoo["zoo_status"],
            },
            "tuna_arrival": {
                "feeding_probability": tuna["feeding_probability"],
                "arrival_window": tuna["estimated_arrival_window"],
                "optimal_date": tuna["optimal_fishing_date"],
                "confidence": tuna["confidence"],
            },
            "recommendation": recommendation,
            "food_chain_score": round(food_chain_score, 3),
            "sst": sst,
            "phi": phi,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
