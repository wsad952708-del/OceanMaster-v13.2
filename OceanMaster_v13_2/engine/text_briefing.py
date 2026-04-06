"""
OceanMaster v13.2 -- Satellite Low-Bandwidth Text Briefing
==========================================================
V3.0 #3: Generate <5KB pure-ASCII daily briefing for satellite transmission.

Design constraints:
  1. Pure ASCII (7-bit safe) -- no Unicode, no emoji, no CJK
  2. Fixed-width 80 columns -- readable on basic terminals
  3. Total size < 5KB (fits Iridium SBD / Inmarsat C)
  4. Self-contained -- no external references needed
  5. Actionable -- Captain can make decisions from this alone

Output format:
  ============================================================
  OCEANMASTER DAILY FISHING BRIEFING
  Date: 2026-02-27 UTC   Valid: 24h
  ============================================================
  TOP HOTSPOTS (5)
  --------------------------------------------------------
  #1  YFT  24.5N 142.3E  HSI=0.82  SST=28.1C  350nm  35h
  #2  BET  22.1N 138.7E  HSI=0.75  SST=24.3C  280nm  28h
  ...
  ============================================================
  SEA CONDITIONS
  SST: 23.5-29.2C   Wind: 8.3 m/s   Current: 0.4 m/s
  ENSO: ONI=+0.30 (Neutral)   Moon: 0.72 (Waning Gibbous)
  ============================================================
  EEZ STATUS
  #1 HIGH_SEAS (OK)   #2 JPN (UNAUTHORIZED - PENALTY)
  ============================================================
  SAFETY
  No typhoon warnings in effect.
  ============================================================

Usage:
  from engine.text_briefing import generate_sat_briefing
  text = generate_sat_briefing(hotspots, summary)
  # len(text.encode('ascii')) < 5000
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("OceanMaster.SatBriefing")

# Species 3-letter codes (ICCAT/WCPFC standard)
SPECIES_CODES = {
    "yellowfin": "YFT",
    "bigeye": "BET",
    "skipjack": "SKJ",
    "albacore": "ALB",
    "swordfish": "SWO",
    "neon_flying_squid": "NFS",
    "japanese_flying_squid": "JFS",
    "squid": "SQD",
}

# Column width for fixed-format
W = 60


def generate_sat_briefing(
    hotspots: List[Dict[str, Any]],
    summary: Optional[Dict[str, Any]] = None,
    max_hotspots: int = 5,
    timestamp: Optional[datetime] = None,
    sea_hazards: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a pure-ASCII satellite briefing.

    Args:
        hotspots: list of hotspot dicts from ai_fusion
        summary: analysis_summary dict from pipeline
        max_hotspots: number of hotspots to include (default 5)
        timestamp: override UTC timestamp

    Returns:
        Pure ASCII string, < 5KB, 80 columns wide
    """
    summary = summary or {}
    ts = timestamp or datetime.now(timezone.utc)
    lines = []

    # ---- Header ----
    lines.append("=" * W)
    lines.append("OCEANMASTER DAILY FISHING BRIEFING".center(W))
    lines.append(f"Date: {ts.strftime('%Y-%m-%d %H:%M')} UTC   Valid: 24h".center(W))
    lines.append("=" * W)

    # ---- Top Hotspots ----
    top = hotspots[:max_hotspots]
    lines.append(f"TOP HOTSPOTS ({len(top)})")
    lines.append("-" * W)
    lines.append(f"{'#':>2} {'SP':3} {'LAT':>6} {'LON':>7} {'HSI':>5} "
                 f"{'SST':>5} {'DIST':>5} {'TIME':>4} {'EEZ':8}")
    lines.append("-" * W)

    for i, h in enumerate(top, 1):
        sp = SPECIES_CODES.get(h.get("species", ""), "???")
        lat = h.get("lat", 0)
        lon = h.get("lon", 0)
        lat_s = f"{abs(lat):.1f}{'N' if lat >= 0 else 'S'}"
        lon_s = f"{abs(lon):.1f}{'E' if lon >= 0 else 'W'}"
        hsi = h.get("hsi", h.get("score", 0))
        sst = h.get("sst")
        sst_s = f"{sst:.1f}" if sst is not None else "  --"
        dist = h.get("distance_nm")
        dist_s = f"{dist:.0f}" if dist is not None else "  --"
        time_h = h.get("travel_hours")
        time_s = f"{time_h:.0f}h" if time_h is not None else " --"

        eez = h.get("eez_code", "---")
        if h.get("eez_authorized") is False:
            eez = eez + "(!)"  # unauthorized marker

        lines.append(
            f"{i:>2} {sp:3} {lat_s:>6} {lon_s:>7} {hsi:5.2f} "
            f"{sst_s:>5} {dist_s:>5} {time_s:>4} {eez:8}"
        )

    # ---- Hotspot Details (brief explanation) ----
    lines.append("")
    for i, h in enumerate(top[:3], 1):
        explain = h.get("explain", h.get("reason_summary", ""))
        if explain:
            explain_ascii = explain.encode("ascii", "replace").decode("ascii")
            lines.append(f"  #{i}: {explain_ascii[:72]}")
        # [海鷹] T100/gradient detail line
        t100 = h.get("t100")
        grad = h.get("gradient_strength")
        dt = h.get("delta_t_surface_100")
        if t100 is not None or grad is not None:
            parts = []
            if t100 is not None:
                parts.append(f"T100={t100:.1f}C")
            if dt is not None:
                parts.append(f"dT={dt:.1f}C")
            if grad is not None:
                parts.append(f"Grad={grad:.3f}C/m")
            lines.append(f"  #{i}: {' | '.join(parts)}")

    # ---- Sea Conditions ----
    lines.append("")
    lines.append("=" * W)
    lines.append("SEA CONDITIONS")
    lines.append("-" * W)

    ssts = [h.get("sst") for h in hotspots if h.get("sst") is not None]
    if ssts:
        lines.append(f"  SST range: {min(ssts):.1f} - {max(ssts):.1f} C")

    # [海鷹] T100 range
    t100s = [h.get("t100") for h in hotspots if h.get("t100") is not None]
    if t100s:
        lines.append(f"  T100 (100m): {min(t100s):.1f} - {max(t100s):.1f} C")
    dts = [h.get("delta_t_surface_100") for h in hotspots if h.get("delta_t_surface_100") is not None]
    if dts:
        lines.append(f"  dT (SST-T100): {min(dts):.1f} - {max(dts):.1f} C")

    ws = summary.get("wind_speed")
    if ws is not None:
        lines.append(f"  Wind:      {ws:.1f} m/s")

    cs = summary.get("current_speed")
    if cs is not None:
        lines.append(f"  Current:   {cs:.1f} m/s")

    oni = summary.get("enso_oni", 0)
    if oni > 0.5:
        enso = "El Nino"
    elif oni < -0.5:
        enso = "La Nina"
    else:
        enso = "Neutral"
    lines.append(f"  ENSO:      ONI={oni:+.2f} ({enso})")

    moon = summary.get("moon_phase")
    if moon is not None:
        if moon < 0.15:
            phase = "New"
        elif moon < 0.35:
            phase = "Waxing Crescent"
        elif moon < 0.65:
            phase = "Full"
        elif moon < 0.85:
            phase = "Waning Gibbous"
        else:
            phase = "Waning Crescent"
        lines.append(f"  Moon:      {moon:.2f} ({phase})")

    # ---- EEZ Summary ----
    eez_issues = [h for h in top if h.get("eez_authorized") is False]
    if eez_issues:
        lines.append("")
        lines.append("=" * W)
        lines.append("EEZ WARNING")
        lines.append("-" * W)
        for h in eez_issues:
            sp = SPECIES_CODES.get(h.get("species", ""), "???")
            lines.append(
                f"  #{h.get('rank', '?')} {sp} in {h.get('eez_code', '?')} "
                f"- UNAUTHORIZED (score penalized 50%)"
            )

    # ---- Safety ----
    lines.append("")
    lines.append("=" * W)
    lines.append("SAFETY ADVISORY")
    lines.append("-" * W)

    typhoon = summary.get("typhoon_warning")
    if typhoon:
        lines.append(f"  *** TYPHOON WARNING: {typhoon} ***")
    else:
        lines.append("  No typhoon or gale warnings in effect.")

    wind_warn = ws is not None and ws > 15.0
    if wind_warn:
        lines.append(f"  HIGH WIND: {ws:.0f} m/s - exercise caution")

    # [v16.0] Sea hazard warnings
    if sea_hazards and sea_hazards.get("has_hazards"):
        for warn in sea_hazards.get("hazard_summary", []):
            lines.append(f"  *** {warn} ***")
        # Rogue wave detail
        if sea_hazards.get("rogue_level") is not None:
            import numpy as _np
            n_rogue_high = int(_np.sum(sea_hazards["rogue_level"] == 2))
            if n_rogue_high > 0:
                lines.append(f"  ROGUE WAVE: {n_rogue_high} cells HIGH risk")
        # Swell detail
        if sea_hazards.get("swell_warning") is not None:
            import numpy as _np
            n_swell = int(_np.sum(sea_hazards["swell_warning"]))
            if n_swell > 0:
                max_dist = float(_np.max(sea_hazards.get("estimated_storm_dist_km", [0])))
                lines.append(f"  SWELL: Tp>16s detected, storm ~{max_dist:.0f}km")
        # Current danger
        if sea_hazards.get("current_danger") is not None:
            import numpy as _np
            n_curr = int(_np.sum(sea_hazards["current_danger"] == 2))
            if n_curr > 0:
                lines.append(f"  STRONG CURRENT: {n_curr} cells >3kn")
    else:
        lines.append("  Sea conditions: normal")

    # ---- Recommended Species ----
    species_set = list({h.get("species", "") for h in top if h.get("species")})
    if species_set:
        sp_codes = [SPECIES_CODES.get(s, s)[:3] for s in species_set]
        lines.append("")
        lines.append(f"RECOMMENDED TARGETS: {', '.join(sp_codes)}")

    # ---- Footer ----
    lines.append("")
    lines.append("=" * W)
    lines.append(f"Source: OceanMaster v13.2 | OISST/VIIRS/CMEMS".center(W))
    lines.append(f"Next update: {(ts.hour + 6) % 24:02d}:00 UTC".center(W))
    lines.append("=" * W)

    text = "\n".join(lines) + "\n"

    # Ensure pure ASCII
    text = text.encode("ascii", "replace").decode("ascii")

    size_bytes = len(text.encode("ascii"))
    log.info(f"Satellite briefing generated: {size_bytes} bytes, "
             f"{len(lines)} lines, {len(top)} hotspots")

    if size_bytes > 5000:
        log.warning(f"Briefing exceeds 5KB target: {size_bytes} bytes")

    return text


def save_briefing_file(
    text: str,
    output_dir: str = "output",
    filename: Optional[str] = None,
) -> str:
    """
    Save briefing to file for satellite transmission.

    Returns:
        Full path to saved file
    """
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now(timezone.utc)
        filename = f"briefing_{ts.strftime('%Y%m%d_%H%M')}.txt"

    path = out / filename
    path.write_text(text, encoding="ascii", errors="replace")
    log.info(f"Briefing saved: {path} ({path.stat().st_size} bytes)")
    return str(path)
