"""
OceanMaster v13.2 — Carbon Emission Calculator [H4]
=====================================================
Calculate CO2 emissions per voyage for ESG/carbon trading compliance.
Formula: CO2 (kg) = fuel_consumption (L) × 3.206 (kg CO2/L diesel)

Ref: IMO Fourth GHG Study (2020), MEPC.1/Circ.684
"""

import logging
from typing import Dict, Any

log = logging.getLogger("OceanMaster.Carbon")

# IMO emission factor for marine diesel (MDO/MGO)
CO2_PER_LITER_DIESEL = 3.206  # kg CO2 per liter


class CarbonCalculator:
    """Calculate voyage CO2 emissions and carbon intensity."""

    def __init__(self, baseline_l_per_nm: float = 45.0):
        self.baseline = baseline_l_per_nm

    def calculate(
        self,
        fuel_consumed_l: float,
        distance_nm: float,
        catch_kg: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a completed or planned voyage.

        Returns:
            {
                "co2_kg": float,
                "co2_tons": float,
                "co2_per_nm": float (kg CO2/nm),
                "co2_per_kg_catch": float (kg CO2/kg fish, if catch > 0),
                "cii_rating": str (A-E),
                "vs_baseline_pct": float (% vs baseline),
                "advisory": str,
            }
        """
        co2_kg = fuel_consumed_l * CO2_PER_LITER_DIESEL
        co2_tons = co2_kg / 1000.0
        co2_per_nm = co2_kg / max(distance_nm, 1.0)
        baseline_co2_per_nm = self.baseline * CO2_PER_LITER_DIESEL
        vs_baseline = (co2_per_nm / baseline_co2_per_nm - 1.0) * 100

        # CII rating (simplified)
        if vs_baseline < -20:
            cii = "A"
        elif vs_baseline < -5:
            cii = "B"
        elif vs_baseline < 10:
            cii = "C"
        elif vs_baseline < 25:
            cii = "D"
        else:
            cii = "E"

        result = {
            "co2_kg": round(co2_kg, 1),
            "co2_tons": round(co2_tons, 2),
            "co2_per_nm": round(co2_per_nm, 2),
            "cii_rating": cii,
            "vs_baseline_pct": round(vs_baseline, 1),
        }

        if catch_kg > 0:
            result["co2_per_kg_catch"] = round(co2_kg / catch_kg, 3)
            result["advisory"] = (
                f"CII={cii}, {co2_tons:.1f}t CO2, "
                f"{co2_kg/catch_kg:.2f} kg CO2/kg catch"
            )
        else:
            result["co2_per_kg_catch"] = 0.0
            result["advisory"] = f"CII={cii}, {co2_tons:.1f}t CO2"

        log.info(f"  Carbon: {co2_tons:.1f}t CO2, CII={cii}")
        return result


class CIIAnnualTracker:
    """
    [v18] Track cumulative CII across multiple voyages for annual compliance.

    IMO CII regulation becomes stricter annually (2023-2030).
    Ref: IMO MEPC.1/Circ.684, ZeroNorth CII optimization concept.

    Usage:
        tracker = CIIAnnualTracker(annual_quota_tons=500)
        tracker.add_voyage(co2_tons=45.2, distance_nm=800)
        status = tracker.get_status()
    """

    def __init__(self, annual_quota_tons_co2: float = 500.0):
        self.quota = annual_quota_tons_co2
        self.voyages = []
        self.total_co2 = 0.0
        self.total_distance = 0.0

    def add_voyage(
        self, co2_tons: float, distance_nm: float = 0.0
    ) -> Dict[str, Any]:
        """Record a completed voyage."""
        self.voyages.append({"co2_tons": co2_tons, "distance_nm": distance_nm})
        self.total_co2 += co2_tons
        self.total_distance += distance_nm

        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Get current annual CII status."""
        remaining = self.quota - self.total_co2
        pct_used = (self.total_co2 / max(self.quota, 1)) * 100

        if remaining < self.quota * 0.1:
            level = "🔴 CRITICAL"
            advice = f"碳排配額僅剩 {remaining:.0f}噸 ({100-pct_used:.0f}%)，建議減速航行"
        elif remaining < self.quota * 0.3:
            level = "🟡 WARNING"
            advice = f"碳排已用 {pct_used:.0f}%，注意控制航速"
        else:
            level = "🟢 NORMAL"
            advice = f"碳排 {pct_used:.0f}%，餘 {remaining:.0f}噸"

        return {
            "total_co2_tons": round(self.total_co2, 1),
            "remaining_tons": round(remaining, 1),
            "pct_used": round(pct_used, 1),
            "voyages_count": len(self.voyages),
            "avg_co2_per_voyage": round(self.total_co2 / max(len(self.voyages), 1), 1),
            "level": level,
            "advice": advice,
        }

