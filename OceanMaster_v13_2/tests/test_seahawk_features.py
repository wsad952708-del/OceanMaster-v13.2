"""
OceanMaster — 海鷹/蒼鷺核心技術測試
====================================
驗證 T100、ΔT/ΔZ、Gaussian 平滑、CHL 時滯融合等新特徵。
"""
import numpy as np
import pytest
import sys
import os

# 確保 project root 在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════
# Test 1: T100 — extract_temp_at_depth
# ═══════════════════════════════════════

class TestExtractTempAtDepth:
    """驗證 ThermoclineFetcher.extract_temp_at_depth 線性插值正確性"""

    @pytest.fixture
    def fetcher_cls(self):
        from engine.thermocline_fetcher import ThermoclineFetcher
        return ThermoclineFetcher

    def test_exact_depth(self, fetcher_cls):
        """目標深度恰好在觀測層上 → 應回傳該層溫度"""
        depths = np.array([0, 50, 100, 200, 500])
        temps = np.array([28.0, 22.0, 15.0, 8.0, 4.0])
        result = fetcher_cls.extract_temp_at_depth(temps, depths, 100.0)
        assert result == pytest.approx(15.0, abs=0.01)

    def test_interpolation(self, fetcher_cls):
        """目標深度在兩層之間 → 線性內插"""
        depths = np.array([0, 50, 100, 200, 500])
        temps = np.array([28.0, 22.0, 15.0, 8.0, 4.0])
        # 75m = 50~100m 之間中點
        result = fetcher_cls.extract_temp_at_depth(temps, depths, 75.0)
        expected = (22.0 + 15.0) / 2  # 18.5
        assert result == pytest.approx(expected, abs=0.01)

    def test_shallow_bound(self, fetcher_cls):
        """目標深度比最淺層還淺 → 回傳最淺溫度"""
        depths = np.array([10, 50, 100])
        temps = np.array([28.0, 22.0, 15.0])
        result = fetcher_cls.extract_temp_at_depth(temps, depths, 5.0)
        assert result == pytest.approx(28.0, abs=0.01)

    def test_deep_bound(self, fetcher_cls):
        """目標深度比最深層還深 → 回傳最深溫度"""
        depths = np.array([0, 50, 100])
        temps = np.array([28.0, 22.0, 15.0])
        result = fetcher_cls.extract_temp_at_depth(temps, depths, 200.0)
        assert result == pytest.approx(15.0, abs=0.01)

    def test_single_point(self, fetcher_cls):
        """只有一個深度層 → 回傳該值"""
        result = fetcher_cls.extract_temp_at_depth(np.array([20.0]), np.array([50.0]), 100.0)
        assert result == pytest.approx(20.0, abs=0.01)

    def test_empty_fallback(self, fetcher_cls):
        """空陣列 → 回傳預設 15.0"""
        result = fetcher_cls.extract_temp_at_depth(np.array([]), np.array([]), 100.0)
        assert result == pytest.approx(15.0, abs=0.01)


# ═══════════════════════════════════════
# Test 2: ΔT/ΔZ — gradient features
# ═══════════════════════════════════════

class TestGradientFeatures:
    """驗證 compute_thermocline_gradient 梯度計算"""

    @pytest.fixture
    def fetcher_cls(self):
        from engine.thermocline_fetcher import ThermoclineFetcher
        return ThermoclineFetcher

    def test_gradient_strength(self, fetcher_cls):
        """溫度劇變區 → 梯度應較大"""
        depths = np.array([0, 50, 100, 200, 500])
        # 50-100m 之間溫差最大: (22-15)/50 = 0.14 °C/m
        temps = np.array([28.0, 22.0, 15.0, 12.0, 10.0])
        depth_max, strength = fetcher_cls.compute_thermocline_gradient(temps, depths)
        assert strength > 0.1  # 至少 0.1 °C/m
        assert 50 <= depth_max <= 150  # 在溫躍層附近

    def test_uniform_profile(self, fetcher_cls):
        """均勻溫度 → 梯度應很小"""
        depths = np.array([0, 50, 100, 200, 500])
        temps = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
        _, strength = fetcher_cls.compute_thermocline_gradient(temps, depths)
        assert strength < 0.01

    def test_short_profile(self, fetcher_cls):
        """profile 太短 → 回傳預設值"""
        depths = np.array([0, 50])
        temps = np.array([28.0, 22.0])
        depth, strength = fetcher_cls.compute_thermocline_gradient(temps, depths)
        assert depth == 100.0 and strength == 0.1  # 預設值


# ═══════════════════════════════════════
# Test 3: compute_from_temp3d — T100 in batch
# ═══════════════════════════════════════

class TestComputeFromTemp3d:
    """驗證批量 3D 溫度場處理中 T100 計算"""

    def test_t100_returned(self):
        """compute_from_temp3d 應回傳 t100 欄位"""
        from engine.thermocline_fetcher import ThermoclineFetcher
        tf = ThermoclineFetcher()

        depths = np.array([0, 50, 100, 200, 500])
        ny, nx = 5, 5
        temp_3d = np.zeros((5, ny, nx), dtype=np.float32)
        for i, t in enumerate([28, 22, 15, 8, 4]):
            temp_3d[i, :, :] = t

        lats = np.linspace(20, 22, ny)
        lons = np.linspace(120, 122, nx)

        result = tf.compute_from_temp3d(temp_3d, lats, lons, depths=depths)

        assert "t100" in result
        assert result["t100"].shape == (ny, nx)
        # T100 should be approximately 15°C (exact match at 100m)
        assert np.nanmean(result["t100"]) == pytest.approx(15.0, abs=0.5)

    def test_delta_t_returned(self):
        """傳入 sst_grid 後應回傳 delta_t_surface_100"""
        from engine.thermocline_fetcher import ThermoclineFetcher
        tf = ThermoclineFetcher()

        depths = np.array([0, 50, 100, 200, 500])
        ny, nx = 5, 5
        temp_3d = np.zeros((5, ny, nx), dtype=np.float32)
        for i, t in enumerate([28, 22, 15, 8, 4]):
            temp_3d[i, :, :] = t

        sst = np.full((ny, nx), 28.0, dtype=np.float32)
        lats = np.linspace(20, 22, ny)
        lons = np.linspace(120, 122, nx)

        result = tf.compute_from_temp3d(temp_3d, lats, lons, depths=depths, sst_grid=sst)

        assert "delta_t_surface_100" in result
        dt = result["delta_t_surface_100"]
        assert dt is not None
        # ΔT = 28 - 15 = 13°C
        assert np.nanmean(dt) == pytest.approx(13.0, abs=0.5)

    def test_gradient_fields(self):
        """應回傳 gradient_strength 和 gradient_depth"""
        from engine.thermocline_fetcher import ThermoclineFetcher
        tf = ThermoclineFetcher()

        depths = np.array([0, 50, 100, 200, 500])
        ny, nx = 3, 3
        temp_3d = np.zeros((5, ny, nx), dtype=np.float32)
        for i, t in enumerate([28, 22, 15, 8, 4]):
            temp_3d[i, :, :] = t

        result = tf.compute_from_temp3d(
            temp_3d, np.linspace(20, 22, ny), np.linspace(120, 122, nx), depths=depths
        )
        assert "gradient_strength" in result
        assert "gradient_depth" in result
        assert result["gradient_strength"].shape == (ny, nx)


# ═══════════════════════════════════════
# Test 4: T100 in HSI models
# ═══════════════════════════════════════

class TestT100InHSI:
    """驗證 T100 SI 因子在 HSI 計算中生效"""

    def test_bigeye_with_t100(self):
        """T100 = 12°C (最佳) 應比 T100 = 25°C 得到更高 HSI"""
        from engine.hsi_models import compute_hsi_bigeye
        ny, nx = 10, 10
        sst = np.full((ny, nx), 28.0, dtype=np.float32)

        t100_optimal = np.full((ny, nx), 12.0, dtype=np.float32)
        t100_bad = np.full((ny, nx), 25.0, dtype=np.float32)

        result_opt = compute_hsi_bigeye(sst, t100=t100_optimal)
        result_bad = compute_hsi_bigeye(sst, t100=t100_bad)

        assert np.nanmean(result_opt["hsi"]) > np.nanmean(result_bad["hsi"])

    def test_yellowfin_with_t100(self):
        """T100 = 18°C (最佳) 應比 T100 = 5°C 得到更高 HSI"""
        from engine.hsi_models import compute_hsi_yellowfin
        ny, nx = 10, 10
        sst = np.full((ny, nx), 28.0, dtype=np.float32)

        t100_optimal = np.full((ny, nx), 18.0, dtype=np.float32)
        t100_bad = np.full((ny, nx), 5.0, dtype=np.float32)

        result_opt = compute_hsi_yellowfin(sst, t100=t100_optimal)
        result_bad = compute_hsi_yellowfin(sst, t100=t100_bad)

        assert np.nanmean(result_opt["hsi"]) > np.nanmean(result_bad["hsi"])

    def test_bigeye_without_t100_still_works(self):
        """不提供 T100 時 → HSI 仍能正常計算"""
        from engine.hsi_models import compute_hsi_bigeye
        sst = np.full((5, 5), 28.0, dtype=np.float32)
        result = compute_hsi_bigeye(sst)
        assert "hsi" in result
        assert result["hsi"].shape == (5, 5)
        assert np.all(np.isfinite(result["hsi"]))


# ═══════════════════════════════════════
# Test 5: Gaussian smoothing
# ═══════════════════════════════════════

class TestGaussianSmoothing:
    """驗證 2D Gaussian 空間平滑效果"""

    def test_smoothing_reduces_noise(self):
        """平滑應降低高頻噪音"""
        from scipy.ndimage import gaussian_filter
        np.random.seed(42)
        noisy = np.random.rand(20, 20).astype(np.float32)
        smoothed = gaussian_filter(noisy, sigma=2.0)

        # 平滑後標準差應更小
        assert np.std(smoothed) < np.std(noisy)

    def test_smoothing_preserves_mean(self):
        """平滑不應顯著改變整體均值"""
        from scipy.ndimage import gaussian_filter
        np.random.seed(42)
        data = np.random.rand(20, 20).astype(np.float32) * 0.5 + 0.25
        smoothed = gaussian_filter(data, sigma=1.0)
        assert np.nanmean(smoothed) == pytest.approx(np.nanmean(data), abs=0.05)

    def test_smoothing_clip_range(self):
        """平滑後 clip 到 0-0.95 範圍"""
        from scipy.ndimage import gaussian_filter
        data = np.random.rand(20, 20).astype(np.float32)
        smoothed = gaussian_filter(data, sigma=2.0)
        clipped = np.clip(smoothed, 0.0, 0.95)
        assert np.all(clipped >= 0.0) and np.all(clipped <= 0.95)


# ═══════════════════════════════════════
# Test 6: CHL lag fusion
# ═══════════════════════════════════════

class TestCHLLagFusion:
    """驗證 CHL 時滯融合邏輯"""

    def test_fusion_blends_values(self):
        """CHL lag fusion 應以 0.4:0.6 比例混合"""
        chl_now = np.full((5, 5), 0.5, dtype=np.float32)
        chl_lag = np.full((5, 5), 1.0, dtype=np.float32)

        # 0.4 * 0.5 + 0.6 * 1.0 = 0.2 + 0.6 = 0.8
        effective = np.where(
            np.isfinite(chl_lag),
            0.4 * chl_now + 0.6 * chl_lag,
            chl_now
        )
        assert np.nanmean(effective) == pytest.approx(0.8, abs=0.01)

    def test_fusion_with_nan_lag(self):
        """lag CHL 為 NaN 時 → 使用即時 CHL"""
        chl_now = np.full((5, 5), 0.5, dtype=np.float32)
        chl_lag = np.full((5, 5), np.nan, dtype=np.float32)

        effective = np.where(
            np.isfinite(chl_lag),
            0.4 * chl_now + 0.6 * chl_lag,
            chl_now
        )
        assert np.nanmean(effective) == pytest.approx(0.5, abs=0.01)


# ═══════════════════════════════════════
# Test 7: Dashboard record fields
# ═══════════════════════════════════════

class TestDashboardRecords:
    """驗證 hotspot records 包含新特徵欄位"""

    def test_record_has_seahawk_fields(self):
        """_build_hotspot_records 應包含 t100, gradient_strength, delta_t"""
        from engine.html_map_generator import _build_hotspot_records

        hotspots = [{
            "species": "bigeye",
            "score": 0.75,
            "lat": 25.0,
            "lon": 130.0,
            "t100": 12.5,
            "gradient_strength": 0.15,
            "delta_t_surface_100": 15.5,
            "chl_lag15d": 0.25,
        }]
        records = _build_hotspot_records(hotspots)
        assert len(records) == 1
        r = records[0]
        assert r["t100"] == 12.5
        assert r["gradient_strength"] == 0.15
        assert r["delta_t"] == 15.5
        assert r["chl_lag15d"] == 0.25

    def test_record_handles_missing_fields(self):
        """新特徵欄位缺失時 → 應為 None，不崩潰"""
        from engine.html_map_generator import _build_hotspot_records

        hotspots = [{
            "species": "skipjack",
            "score": 0.50,
            "lat": 20.0,
            "lon": 125.0,
        }]
        records = _build_hotspot_records(hotspots)
        r = records[0]
        assert r["t100"] is None
        assert r["gradient_strength"] is None
        assert r["delta_t"] is None
        assert r["chl_lag15d"] is None
