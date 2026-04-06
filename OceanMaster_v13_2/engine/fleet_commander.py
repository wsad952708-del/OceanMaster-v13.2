"""
OceanMaster v13.2 — PSO Fleet Commander [H1]
==============================================
Multi-vessel cooperative assignment using Particle Swarm Optimization.
Divides high-HSI zones among N vessels to avoid intra-fleet competition.

Ref: Kennedy & Eberhart (1995) "Particle Swarm Optimization" IEEE ICNN.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional

log = logging.getLogger("OceanMaster.Fleet")


class FleetCommander:
    """Assign fishing zones to fleet vessels using PSO optimization."""

    def __init__(self, n_particles: int = 30, n_iterations: int = 50):
        self.n_particles = n_particles
        self.n_iters = n_iterations

    def assign_zones(
        self,
        hotspots: List[Dict[str, Any]],
        n_vessels: int,
        vessel_positions: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Assign hotspots to vessels to maximize total fleet HSI
        while minimizing overlap and transit distance.

        Args:
            hotspots: list of hotspot dicts with lat, lon, score
            n_vessels: number of vessels in fleet
            vessel_positions: [{lat, lon, name}, ...] current positions

        Returns:
            {
                "assignments": [{vessel_id, hotspots, total_score, zone_center}, ...],
                "total_fleet_score": float,
                "overlap_penalty": float,
            }
        """
        if n_vessels <= 0 or not hotspots:
            return {"assignments": [], "total_fleet_score": 0.0, "overlap_penalty": 0.0}

        n_hs = len(hotspots)
        if n_vessels >= n_hs:
            # One-to-one assignment
            assignments = []
            for i, h in enumerate(hotspots[:n_vessels]):
                assignments.append({
                    "vessel_id": i,
                    "vessel_name": (vessel_positions[i]["name"]
                                    if vessel_positions and i < len(vessel_positions)
                                    else f"Vessel_{i+1}"),
                    "hotspots": [h],
                    "total_score": h.get("score", 0),
                    "zone_center": {"lat": h["lat"], "lon": h["lon"]},
                })
            return {
                "assignments": assignments,
                "total_fleet_score": sum(a["total_score"] for a in assignments),
                "overlap_penalty": 0.0,
            }

        # PSO to find optimal assignment
        # Encoding: each particle = [n_hs] integers in [0, n_vessels)
        rng = np.random.default_rng(42)
        coords = np.array([[h["lat"], h["lon"]] for h in hotspots])
        scores = np.array([h.get("score", 0.5) for h in hotspots])

        # Initialize particles
        particles = rng.integers(0, n_vessels, (self.n_particles, n_hs))
        velocities = rng.uniform(-1, 1, (self.n_particles, n_hs))
        pbest = particles.copy()
        pbest_scores = np.array([self._fitness(p, coords, scores, n_vessels) for p in particles])
        gbest_idx = np.argmax(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        w, c1, c2 = 0.7, 1.5, 1.5
        for it in range(self.n_iters):
            for i in range(self.n_particles):
                r1, r2 = rng.random(n_hs), rng.random(n_hs)
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (pbest[i] - particles[i])
                                 + c2 * r2 * (gbest - particles[i]))
                particles[i] = np.clip(
                    np.round(particles[i] + velocities[i]).astype(int),
                    0, n_vessels - 1)
                fit = self._fitness(particles[i], coords, scores, n_vessels)
                if fit > pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = fit
                if fit > gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = fit

        # Build assignments from gbest
        assignments = []
        for v in range(n_vessels):
            mask = gbest == v
            v_hotspots = [hotspots[j] for j in range(n_hs) if mask[j]]
            v_score = float(np.sum(scores[mask])) if np.any(mask) else 0.0
            v_center = {"lat": float(np.mean(coords[mask, 0])) if np.any(mask) else 0,
                        "lon": float(np.mean(coords[mask, 1])) if np.any(mask) else 0}
            assignments.append({
                "vessel_id": v,
                "vessel_name": (vessel_positions[v]["name"]
                                if vessel_positions and v < len(vessel_positions)
                                else f"Vessel_{v+1}"),
                "hotspots": v_hotspots,
                "n_hotspots": len(v_hotspots),
                "total_score": round(v_score, 3),
                "zone_center": v_center,
            })

        total = sum(a["total_score"] for a in assignments)
        log.info(f"  Fleet PSO: {n_vessels} vessels, {n_hs} hotspots, "
                 f"total_score={total:.2f}")

        return {
            "assignments": assignments,
            "total_fleet_score": round(total, 3),
            "overlap_penalty": round(float(gbest_score - total), 3),
        }

    def _fitness(self, assignment, coords, scores, n_vessels):
        """Fitness = total score - overlap penalty."""
        total = 0.0
        penalty = 0.0
        for v in range(n_vessels):
            mask = assignment == v
            if not np.any(mask):
                continue
            total += np.sum(scores[mask])
            # Penalize if vessel's hotspots are too spread out
            v_coords = coords[mask]
            if len(v_coords) > 1:
                spread = np.std(v_coords, axis=0).sum()
                penalty += spread * 0.1
        return total - penalty
