"""
OceanMaster v13.2 — Acoustic Scattering Layer Proxy [F1]
==========================================================
Estimate deep scattering layer (DSL) biomass from oceanographic variables.
Proxy for micronekton abundance without acoustic survey data.

Ref: Irigoien et al. (2014) Nature Comm. 5:3271
     Proud et al. (2017) ICES J. Marine Sci. 74:2006-2015
"""

import numpy as np
import logging
from typing import Dict, Any

log = logging.getLogger("OceanMaster.Acoustic")


class AcousticScatteringProxy:
    """
    [v16.0 F1] Estimate deep scattering layer (DSL) biomass.

    The DSL contains micronekton (2-20cm organisms) that undergo
    diel vertical migration (DVM). This proxy estimates their biomass
    from surface oceanographic variables.

    Key relationships:
    - NPP drives base biomass
    - MLD affects access to surface production
    - OMZ depth compresses habitat
    - Lunar phase affects DVM amplitude
    """

    def __init__(self):
        pass

    def estimate(
        self,
        npp: float = 300.0,         # mg C/m²/day
        mld: float = 50.0,          # mixed layer depth (m)
        omz_depth: float = 400.0,   # oxygen minimum zone depth (m)
        sst: float = 27.0,          # SST (°C)
        lunar_illumination: float = 0.5,  # 0-1
    ) -> Dict[str, Any]:
        """
        Estimate DSL parameters.

        Returns:
            {
                "dsl_biomass_index": float (0-1, relative),
                "dsl_depth_day": float (m, daytime center),
                "dsl_depth_night": float (m, nighttime center),
                "dvm_amplitude": float (m, migration distance),
                "scattering_strength_db": float (proxy Sv),
                "advisory": str,
            }
        """
        # NPP → base biomass (log relationship)
        npp_factor = np.clip(np.log10(max(npp, 1)) / np.log10(1000), 0, 1)

        # MLD effect: shallow MLD → more accessible production
        mld_factor = np.clip(1.0 - mld / 200.0, 0.2, 1.0)

        # OMZ compression: shallow OMZ compresses DSL,
        # increasing density at edges
        omz_factor = np.clip(1.0 + (500 - omz_depth) / 500, 0.5, 1.5)

        # Temperature: warmer = higher metabolic rates = more biomass turnover
        temp_factor = np.clip((sst - 15) / 15, 0.3, 1.2)

        biomass_index = float(npp_factor * mld_factor * omz_factor * temp_factor)
        biomass_index = min(1.0, max(0.0, biomass_index))

        # DSL depth: typically 300-600m daytime
        dsl_day = 400.0 + 100.0 * (1 - npp_factor)  # deeper when less food
        dsl_night = max(20.0, dsl_day * 0.15)  # migrate to ~15% of day depth

        # DVM amplitude affected by lunar illumination
        # Full moon → shallower migration (less darkness at surface)
        dvm_amplitude = dsl_day - dsl_night
        dvm_amplitude *= (1.0 - 0.3 * lunar_illumination)

        dsl_night_adjusted = dsl_day - dvm_amplitude

        # Scattering strength proxy (dB, typical -80 to -50 Sv)
        sv = -80.0 + 30.0 * biomass_index

        if biomass_index > 0.7:
            advisory = "DSL biomass HIGH - strong prey availability"
        elif biomass_index > 0.4:
            advisory = "DSL biomass moderate"
        else:
            advisory = "DSL biomass low - limited prey base"

        log.info(f"  DSL: biomass={biomass_index:.2f}, "
                 f"day={dsl_day:.0f}m, night={dsl_night_adjusted:.0f}m, "
                 f"DVM={dvm_amplitude:.0f}m")

        return {
            "dsl_biomass_index": round(biomass_index, 3),
            "dsl_depth_day": round(dsl_day, 1),
            "dsl_depth_night": round(float(dsl_night_adjusted), 1),
            "dvm_amplitude": round(float(dvm_amplitude), 1),
            "scattering_strength_db": round(sv, 1),
            "advisory": advisory,
        }
