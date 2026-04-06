"""
OceanMaster v13.2 — Market Price Optimizer [H3]
=================================================
Optimize target species selection based on seasonal market prices
and predicted catch composition for maximum revenue.

Ref: FAO FishStat / Taiwan Fisheries Agency price data
"""

import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.Market")

# Average auction price (TWD/kg) by species and season
# Source: 前鎮漁港拍賣價格 2020-2024 average
SEASONAL_PRICES = {
    "yellowfin": {
        "prices": [280, 260, 250, 240, 220, 200, 190, 200, 230, 260, 290, 300],  # Jan-Dec
        "unit": "TWD/kg",
    },
    "bigeye": {
        "prices": [350, 340, 320, 300, 280, 260, 250, 260, 300, 330, 360, 380],
        "unit": "TWD/kg",
    },
    "skipjack": {
        "prices": [60, 55, 50, 45, 40, 35, 35, 40, 50, 55, 60, 65],
        "unit": "TWD/kg",
    },
    "albacore": {
        "prices": [180, 170, 160, 150, 140, 130, 130, 140, 160, 170, 180, 190],
        "unit": "TWD/kg",
    },
    "swordfish": {
        "prices": [220, 210, 200, 180, 170, 160, 160, 170, 190, 210, 220, 230],
        "unit": "TWD/kg",
    },
}


class MarketOptimizer:
    """[v16.0 H3] Optimize species targeting for maximum revenue."""

    def __init__(self, custom_prices: Optional[Dict] = None):
        self.prices = custom_prices or SEASONAL_PRICES

    def optimize(
        self,
        catch_composition: Dict[str, float],
        target_month: Optional[int] = None,
        estimated_catch_kg: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        Calculate expected revenue and optimal targeting strategy.

        Args:
            catch_composition: {"species": probability%} from CatchCompositionForecaster
            target_month: 1-12, default=current month
            estimated_catch_kg: estimated total catch (kg)

        Returns:
            {
                "revenue_by_species": dict,
                "total_expected_revenue": float (TWD),
                "best_target_species": str,
                "price_per_kg": dict,
                "revenue_advisory": str,
            }
        """
        month = target_month or datetime.now().month
        month_idx = month - 1

        revenue = {}
        price_map = {}

        for species, prob_pct in catch_composition.items():
            if species not in self.prices:
                continue
            price_kg = self.prices[species]["prices"][month_idx]
            catch_kg = estimated_catch_kg * prob_pct / 100.0
            rev = price_kg * catch_kg
            revenue[species] = round(rev, 0)
            price_map[species] = price_kg

        total_revenue = sum(revenue.values())

        # Revenue per effort: which species gives best return?
        rev_per_prob = {}
        for sp, rev in revenue.items():
            prob = catch_composition.get(sp, 0)
            if prob > 0:
                rev_per_prob[sp] = rev / prob  # revenue per % probability
        best_target = max(rev_per_prob, key=rev_per_prob.get) if rev_per_prob else "unknown"

        # Find which month has best price for dominant species
        if best_target in self.prices:
            prices_all = self.prices[best_target]["prices"]
            best_month = int(np.argmax(prices_all)) + 1
            current_price = prices_all[month_idx]
            peak_price = max(prices_all)
            price_ratio = current_price / peak_price * 100
        else:
            best_month = month
            price_ratio = 100

        advisory = (f"最佳目標: {best_target} @ {price_map.get(best_target, 0)} TWD/kg | "
                    f"預估收入: {total_revenue:,.0f} TWD | "
                    f"{'當月高價位' if price_ratio > 85 else f'最佳月份: {best_month}月'}")

        return {
            "revenue_by_species": revenue,
            "total_expected_revenue": total_revenue,
            "best_target_species": best_target,
            "price_per_kg": price_map,
            "price_ratio_pct": round(price_ratio, 1),
            "best_price_month": best_month,
            "revenue_advisory": advisory,
        }
