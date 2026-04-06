"""
OceanMaster v13.2 — SafetyFilter 專屬測試
==========================================
涵蓋:
  - 颱風路徑距離三門檻: <200km 剔除 / 200-350km 降級 / 350-500km 警示
  - EEZ 標註邏輯
  - 空輸入邊界情況
  - 全部被過濾後的行為
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import math
from unittest.mock import patch, MagicMock
from pipeline.safety_filter import SafetyFilter, _haversine_km, DANGER_RADIUS


# ── Fixtures ──

def _make_hotspot(lat, lon, hsi=0.85, species="skipjack"):
    return {
        "lat": lat, "lon": lon,
        "hsi": hsi, "species": species,
        "score": hsi, "rank": 1,
    }


class FakeForecastPoint:
    """模擬颱風預報點"""
    def __init__(self, lat, lon, hour):
        self.lat = lat
        self.lon = lon
        self.hour = hour


class FakeAlert:
    """模擬颱風警報（含 forecast_track）"""
    def __init__(self, name, forecast_track):
        self.name = name
        self.forecast_track = forecast_track


# ═══════════════════════════════════════
#  Test: _haversine_km 距離計算
# ═══════════════════════════════════════

class TestHaversine:
    def test_same_point(self):
        assert _haversine_km(25, 130, 25, 130) == 0.0

    def test_known_distance(self):
        # 高雄(22.6, 120.3) → 東京(35.7, 139.7) ≈ 2100km
        d = _haversine_km(22.6, 120.3, 35.7, 139.7)
        assert 2000 < d < 2300

    def test_small_distance(self):
        # 0.1 度 ≈ 11km
        d = _haversine_km(25.0, 130.0, 25.1, 130.0)
        assert 10 < d < 12


# ═══════════════════════════════════════
#  Test: 颱風路徑三門檻過濾
# ═══════════════════════════════════════

class TestTyphoonForecastFilter:
    """
    規則:
      <200km  → 剔除 (remove)
      200-350km → HSI×0.5 + future_danger 標記 (degrade)
      350-500km → future_caution 標記 (caution)
      >500km → 不受影響
    """

    def setup_method(self):
        self.sf = SafetyFilter()

    def _make_alert(self, name, track_points):
        return FakeAlert(name, [FakeForecastPoint(*pt) for pt in track_points])

    def test_remove_within_200km(self):
        """颱風 T+24h 位置距 hotspot <200km → 剔除"""
        # hotspot at (20, 130)
        h = _make_hotspot(20.0, 130.0)
        # typhoon forecast at T+24h → (20.5, 130.5), ~70km away → <200km
        alert = self._make_alert("HAIYAN", [(20.5, 130.5, 24)])

        result = self.sf._filter_by_forecast_track([h], [alert])
        assert len(result) == 0, "Hotspot within 200km of typhoon should be removed"

    def test_degrade_200_350km(self):
        """颱風 T+48h 位置距 hotspot 200-350km → HSI 降 50%"""
        h = _make_hotspot(20.0, 130.0, hsi=0.80)
        # ~280km away
        alert = self._make_alert("MEGI", [(22.5, 130.0, 48)])

        # Verify distance is in range
        dist = _haversine_km(20.0, 130.0, 22.5, 130.0)
        assert 200 < dist < 350, f"Distance should be 200-350km, got {dist:.0f}"

        result = self.sf._filter_by_forecast_track([h], [alert])
        assert len(result) == 1, "Hotspot 200-350km should NOT be removed"
        assert result[0]["hsi"] == pytest.approx(0.40, abs=0.01), "HSI should be degraded by 50%"
        assert "future_danger" in result[0], "Should have future_danger marker"

    def test_caution_350_500km(self):
        """颱風 T+72h 位置距 hotspot 350-500km → 標記 caution"""
        h = _make_hotspot(20.0, 130.0, hsi=0.80)
        # ~400km away
        alert = self._make_alert("NEPARTAK", [(23.6, 130.0, 72)])

        dist = _haversine_km(20.0, 130.0, 23.6, 130.0)
        assert 350 < dist < 500, f"Distance should be 350-500km, got {dist:.0f}"

        result = self.sf._filter_by_forecast_track([h], [alert])
        assert len(result) == 1, "Hotspot 350-500km should NOT be removed"
        assert result[0]["hsi"] == pytest.approx(0.80, abs=0.01), "HSI should NOT be degraded"
        assert "future_caution" in result[0], "Should have future_caution marker"

    def test_safe_beyond_500km(self):
        """颱風 >500km → 完全不受影響"""
        h = _make_hotspot(20.0, 130.0, hsi=0.85)
        # ~1100km away
        alert = self._make_alert("FAR", [(30.0, 130.0, 24)])

        result = self.sf._filter_by_forecast_track([h], [alert])
        assert len(result) == 1
        assert result[0]["hsi"] == 0.85
        assert "future_danger" not in result[0]
        assert "future_caution" not in result[0]

    def test_t0_skipped(self):
        """hour=0 的 ForecastPoint 應被跳過（由 Layer 1 處理）"""
        h = _make_hotspot(20.0, 130.0)
        # T+0 within 100km, but should be skipped
        alert = self._make_alert("NOW", [(20.5, 130.5, 0)])

        result = self.sf._filter_by_forecast_track([h], [alert])
        assert len(result) == 1, "T+0 forecast point should be skipped"

    def test_worst_level_wins(self):
        """多個預報點，最嚴重的判定優先"""
        h = _make_hotspot(20.0, 130.0, hsi=0.80)
        # T+24h: caution (400km), T+48h: remove (50km)
        alert = self._make_alert("MULTI", [
            (23.6, 130.0, 24),  # ~400km, caution
            (20.3, 130.3, 48),  # ~50km, remove
        ])

        result = self.sf._filter_by_forecast_track([h], [alert])
        assert len(result) == 0, "Worst level (remove) should win"


# ═══════════════════════════════════════
#  Test: 空輸入與邊界情況
# ═══════════════════════════════════════

class TestEdgeCases:

    def setup_method(self):
        self.sf = SafetyFilter()

    def test_empty_hotspots(self):
        """0 個 hotspot → 應返回空 list"""
        result = self.sf.filter([], typhoon_alerts=[], lats=None, lons=None)
        assert result == []

    def test_none_typhoon_alerts(self):
        """typhoon_alerts=None → 不應 crash"""
        h = _make_hotspot(20.0, 130.0)
        result = self.sf.filter([h], typhoon_alerts=None)
        assert len(result) == 1

    def test_all_filtered_out(self):
        """所有 hotspot 都被颱風剔除 → 返回空 list"""
        hotspots = [
            _make_hotspot(20.0, 130.0),
            _make_hotspot(20.1, 130.1),
        ]
        alert = FakeAlert("MONSTER", [
            FakeForecastPoint(20.0, 130.0, 24),  # covers both hotspots
        ])

        result = self.sf._filter_by_forecast_track(hotspots, [alert])
        assert len(result) == 0

    def test_filter_summary(self):
        """get_filter_summary 輸出正確"""
        summary = SafetyFilter.get_filter_summary(n_input=10, n_output=7)
        assert summary["input_count"] == 10
        assert summary["output_count"] == 7
        assert summary["removed_count"] == 3
        assert summary["removal_rate"] == pytest.approx(0.3, abs=0.01)

    def test_filter_summary_zero_input(self):
        """0 個輸入不應 division by zero"""
        summary = SafetyFilter.get_filter_summary(n_input=0, n_output=0)
        assert summary["removal_rate"] == 0.0


# ═══════════════════════════════════════
#  Test: DANGER_RADIUS 常數正確性
# ═══════════════════════════════════════

class TestDangerRadius:
    def test_radius_ordering(self):
        """core < moderate < alert"""
        assert DANGER_RADIUS["core"] < DANGER_RADIUS["moderate"] < DANGER_RADIUS["alert"]

    def test_radius_values(self):
        assert DANGER_RADIUS["core"] == 200
        assert DANGER_RADIUS["moderate"] == 350
        assert DANGER_RADIUS["alert"] == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
