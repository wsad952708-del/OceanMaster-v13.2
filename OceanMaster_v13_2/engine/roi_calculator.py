"""
OceanMaster v13.2 — ROI Calculator
==================================
將技術指標轉換成船東看得懂的「新台幣節省數字」。
靈感: Catchwise (A/B 測試 +5.8% 效率提升)

Ref: 漁業署遠洋漁業統計年報 (2024)

Usage:
    roi = ROICalculator()
    report = roi.calculate(
        distance_saved_nm=50, fuel_saved_l=2000,
        time_saved_hours=8, catch_increase_pct=5.0,
    )
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

log = logging.getLogger("OceanMaster.ROI")


class ROICalculator:
    """Convert technical fishing metrics to financial ROI."""

    # Default market prices (TWD)
    DIESEL_PRICE_TWD = 28.5       # NT$/L (2025 avg marine diesel)
    CREW_HOURLY_COST = 2500       # NT$/hr (全船人事+折舊+保險)
    FISH_PRICES_TWD = {           # NT$/kg (產地價)
        "yellowfin":  180,
        "bigeye":     250,
        "skipjack":   90,
        "albacore":   200,
        "swordfish":  350,
        "squid":      120,
        "japanese_flying_squid": 150,
    }
    AVG_CATCH_KG_PER_TRIP = 50000  # 50 噸/航次 (延繩釣平均)

    def __init__(
        self,
        diesel_price: Optional[float] = None,
        fish_prices: Optional[Dict[str, float]] = None,
    ):
        if diesel_price:
            self.DIESEL_PRICE_TWD = diesel_price
        if fish_prices:
            self.FISH_PRICES_TWD.update(fish_prices)

    def calculate(
        self,
        distance_saved_nm: float = 0.0,
        fuel_saved_l: float = 0.0,
        time_saved_hours: float = 0.0,
        catch_increase_pct: float = 0.0,
        species: str = "yellowfin",
        avg_catch_kg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate trip-level ROI.

        Args:
            distance_saved_nm: 節省的航程 (海浬)
            fuel_saved_l: 節省的油料 (公升)
            time_saved_hours: 節省的時間 (小時)
            catch_increase_pct: 漁獲增加比例 (%)
            species: 主要物種
            avg_catch_kg: 平均漁獲量 (kg), None=用預設

        Returns:
            ROI report dict with TWD savings breakdown
        """
        catch_kg = avg_catch_kg or self.AVG_CATCH_KG_PER_TRIP
        fish_price = self.FISH_PRICES_TWD.get(species, 180)

        # 1. 油料節省
        fuel_savings = fuel_saved_l * self.DIESEL_PRICE_TWD

        # 2. 時間節省 (人事+折舊+保險)
        time_savings = time_saved_hours * self.CREW_HOURLY_COST

        # 3. 漁獲增加收入
        catch_bonus_kg = catch_kg * (catch_increase_pct / 100)
        catch_bonus_twd = catch_bonus_kg * fish_price

        # 合計
        total = fuel_savings + time_savings + catch_bonus_twd
        annual = total * 12  # 假設月出航一次

        result = {
            "fuel_savings_twd": round(fuel_savings),
            "time_savings_twd": round(time_savings),
            "catch_bonus_twd": round(catch_bonus_twd),
            "total_savings_twd": round(total),
            "annual_projection_twd": round(annual),
            "fuel_saved_l": round(fuel_saved_l, 1),
            "time_saved_hours": round(time_saved_hours, 1),
            "catch_increase_pct": round(catch_increase_pct, 1),
            "catch_bonus_kg": round(catch_bonus_kg, 1),
            "species": species,
            "fish_price_twd": fish_price,
            "pitch": (
                f"每航次節省 NT${total:,.0f} "
                f"(油 NT${fuel_savings:,.0f} + 時間 NT${time_savings:,.0f} "
                f"+ 增產 NT${catch_bonus_twd:,.0f})，"
                f"年省 NT${annual:,.0f}"
            ),
        }

        log.info(f"  💰 ROI: {result['pitch']}")
        return result

    def compare_routes(
        self,
        route_a_dist_nm: float,
        route_b_dist_nm: float,
        fuel_l_per_nm: float = 45.0,
        speed_kn: float = 10.0,
        species: str = "yellowfin",
    ) -> Dict[str, Any]:
        """Compare two routes and return savings."""
        dist_diff = route_a_dist_nm - route_b_dist_nm
        fuel_diff = dist_diff * fuel_l_per_nm
        time_diff = dist_diff / max(speed_kn, 1.0)

        return self.calculate(
            distance_saved_nm=max(0, dist_diff),
            fuel_saved_l=max(0, fuel_diff),
            time_saved_hours=max(0, time_diff),
            species=species,
        )

    def generate_report(
        self,
        roi: Dict[str, Any],
        vessel_name: str = "OceanMaster 用戶",
    ) -> str:
        """Generate human-readable ROI report."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"""
╔══════════════════════════════════════════════╗
║  OceanMaster ROI 報告  —  {now}
╠══════════════════════════════════════════════╣
║  船舶: {vessel_name}
║  目標物種: {roi.get('species', 'yellowfin')}
╠══════════════════════════════════════════════╣
║  ⛽ 油料節省:    NT$ {roi.get('fuel_savings_twd', 0):>10,}
║  ⏱️ 時間節省:    NT$ {roi.get('time_savings_twd', 0):>10,}
║  🐟 漁獲增產:    NT$ {roi.get('catch_bonus_twd', 0):>10,}
╠══════════════════════════════════════════════╣
║  📊 單航次節省:  NT$ {roi.get('total_savings_twd', 0):>10,}
║  📈 年度節省:    NT$ {roi.get('annual_projection_twd', 0):>10,}
╚══════════════════════════════════════════════╝
"""
