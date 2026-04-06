"""
OceanMaster v12 — 氧最小層 (OMZ) 棲息壓縮模型  # [v12-enhance]
================================================
偵測溶氧最小層 (OMZ) 並計算對鮪魚垂直棲息空間的壓縮效應。

科學背景:
  - OMZ 定義: DO < 3.5 ml/L (Stramma 2008)
  - 當 OMZ 較淺時，大目鮪和黃鰭鮪的垂直棲息深度被壓縮到更淺層
  - 壓縮嚴重 → 魚聚集在更薄的水層 → 表層密度可能增加但整體棲息品質下降
  - 文獻: Stramma et al. 2012 (Nature CC), Mislan et al. 2017

v12 新增:
  - [v12-enhance] OMZ 邊緣富集效應 (Stramma et al. 2012)
    OMZ 邊緣 = 高 DO 梯度區 → 反硝化作用提供營養鹽 → 餌料聚集
  - [v12-enhance] 合併棲息壓縮 + 邊緣富集的淨效應指數

數據源:
  - WOA2023 溶氧剖面 (dissolved_oxygen.py 已有接口)
  - CMEMS BGC o2 (若有 API key)
"""

import numpy as np
import logging
from typing import Dict, Optional

log = logging.getLogger("OceanMaster.OMZ")

# 物種臨界溶氧 (ml/L) — Stramma 2012 + Mislan 2017
SPECIES_PCRIT_MLL = {
    "skipjack":  3.5,   # 淺層物種，對低氧最敏感
    "yellowfin": 3.0,   # 中等耐受
    "bigeye":    2.0,   # 深潛物種，耐低氧能力最強
    "albacore":  2.5,   # 中深層活動
}

# 物種最大潛水深度 (m) — 作為 OMZ 壓縮的基準
SPECIES_MAX_DEPTH = {
    "skipjack":  250,
    "yellowfin": 300,
    "bigeye":    700,
    "albacore":  400,
}

# WOA 標準深度 (m)
WOA_DEPTHS = np.array([
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
    85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350,
    375, 400, 450, 500, 550, 600, 700, 800, 900, 1000
])


class OMZModel:
    """
    氧最小層偵測與棲息壓縮計算

    使用:
        omz = OMZModel()
        omz_top = omz.find_omz_top(do_3d, depths, pcrit_mll=3.5)
        compression = omz.compute_habitat_compression('bigeye', omz_top_m=180, z20=300)
    """

    def find_omz_top(
        self,
        do_3d: np.ndarray,
        depths: np.ndarray,
        pcrit_mll: float = 3.5,
    ) -> np.ndarray:
        """
        找到 DO < Pcrit 的最淺深度 → 棲息上限

        Parameters:
            do_3d: shape (n_depths, n_lat, n_lon) — 3D 溶氧場 (ml/L)
            depths: shape (n_depths,) — 深度陣列 (m)
            pcrit_mll: 臨界溶氧閾值 (ml/L)

        Returns:
            omz_top: shape (n_lat, n_lon) — OMZ 頂部深度 (m)
                     超出深度陣列範圍 → 設為最大深度+100 (無 OMZ)
        """
        n_depths, n_lat, n_lon = do_3d.shape
        omz_top = np.full((n_lat, n_lon), depths[-1] + 100.0)  # 預設: 無 OMZ

        for iy in range(n_lat):
            for ix in range(n_lon):
                profile = do_3d[:, iy, ix]
                # 找第一個 DO < Pcrit 的深度
                for iz in range(n_depths):
                    if np.isfinite(profile[iz]) and profile[iz] < pcrit_mll:
                        # 線性內插: 精確找到 DO = Pcrit 的深度
                        if iz > 0 and np.isfinite(profile[iz - 1]):
                            do_above = profile[iz - 1]
                            do_below = profile[iz]
                            if do_above > pcrit_mll and do_above != do_below:
                                frac = (do_above - pcrit_mll) / (do_above - do_below)
                                omz_top[iy, ix] = depths[iz - 1] + frac * (depths[iz] - depths[iz - 1])
                            else:
                                omz_top[iy, ix] = float(depths[iz])
                        else:
                            omz_top[iy, ix] = float(depths[iz])
                        break

        return omz_top.astype(np.float32)

    def compute_habitat_compression(
        self,
        species: str,
        omz_top_m: "np.ndarray | float",
        z20: "np.ndarray | float" = 200.0,
    ) -> "np.ndarray | float":
        """
        棲息壓縮指數 (0-1)

        原理 (Prince & Goodyear 2006, Stramma et al. 2012):
          - 魚的垂直空間上界 = 表面溫度 (MLD)
          - 魚的垂直空間下界 = min(species_max_depth, OMZ top)
          - 壓縮比 = actual_range / potential_range
          - 壓縮嚴重 (指數 < 0.3) → 某些區域魚被迫聚集在薄層

        Parameters:
            species: 物種 ID
            omz_top_m: OMZ 頂部深度 (m)，可為標量或 2D 陣列
            z20: 溫躍層深度 (m)

        Returns:
            compression_index: 0-1 (1=無壓縮，0=完全壓縮)
        """
        max_depth = SPECIES_MAX_DEPTH.get(species, 400)

        # 有效棲息深度 = min(物種最大深度, OMZ 頂部)
        if isinstance(omz_top_m, (int, float)):
            effective_depth = min(max_depth, float(omz_top_m))
            potential_depth = float(max_depth)
        else:
            effective_depth = np.minimum(max_depth, omz_top_m)
            potential_depth = float(max_depth)

        # 壓縮指數 = 實際可用深度 / 潛在最大深度
        compression = effective_depth / potential_depth

        if isinstance(compression, np.ndarray):
            compression = np.clip(compression, 0.0, 1.0).astype(np.float32)
        else:
            compression = max(0.0, min(1.0, compression))

        return compression

    def compute_compression_for_all_species(
        self,
        do_3d: np.ndarray,
        depths: np.ndarray,
        z20: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        計算所有物種的棲息壓縮

        Returns:
            dict: {species: compression_index_2d}
        """
        results = {}
        for species, pcrit in SPECIES_PCRIT_MLL.items():
            omz_top = self.find_omz_top(do_3d, depths, pcrit)
            z20_val = z20 if z20 is not None else 200.0
            compression = self.compute_habitat_compression(species, omz_top, z20_val)
            results[species] = compression

            if isinstance(compression, np.ndarray):
                log.info(f"  {species}: OMZ top = {np.nanmean(omz_top):.0f}m, "
                         f"compression = {np.nanmean(compression):.2f}")
            else:
                log.info(f"  {species}: compression = {compression:.2f}")

        return results

    # ── [v12-enhance] OMZ 邊緣富集效應 ──

    def compute_omz_edge_enrichment(
        self,
        do_3d: np.ndarray,
        depths: np.ndarray,
        target_depth_idx: int = 18,  # ~100m WOA 標準深度
    ) -> np.ndarray:
        """
        [v12-enhance] OMZ 邊緣富集指數 (Stramma et al. 2012, Nature CC)

        OMZ 邊緣（高 DO 梯度區域）有以下生態效應:
          1. 反硝化作用 (denitrification) → 釋放 N₂O 和再生營養鹽
          2. DO 梯度 → 垂直混合增強 → 營養鹽上湧
          3. 餌料生物聚集在 OMZ 邊緣（可呼吸的最深處）

        結果: OMZ 邊緣 = 天然漁場 (尤其對大目鮪和黃鰭鮪)

        Parameters:
            do_3d: shape (n_depths, n_lat, n_lon) — 3D 溶氧場
            depths: shape (n_depths,) — 深度陣列
            target_depth_idx: 計算水平DO梯度的深度層索引

        Returns:
            edge_enrichment: 2D (0-1) 邊緣富集指數
        """
        # 取目標深度的 DO 水平分布
        if target_depth_idx >= do_3d.shape[0]:
            target_depth_idx = do_3d.shape[0] - 1

        do_layer = do_3d[target_depth_idx]  # 2D (lat, lon)

        # 計算水平 DO 梯度 → |∇DO|
        grad_y, grad_x = np.gradient(np.nan_to_num(do_layer, nan=5.0))
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)

        # 歸一化梯度
        p95 = np.nanpercentile(grad_mag, 95)
        if p95 > 0:
            grad_norm = np.clip(grad_mag / p95, 0, 1)
        else:
            grad_norm = np.zeros_like(do_layer)

        # 只在低氧區邊緣有效 (DO 在 2-5 ml/L 範圍內)
        # OMZ 中心 (DO < 1.5) → 太缺氧，不適合
        # 遠離 OMZ (DO > 5) → 沒有富集效應
        edge_mask = np.where(
            (do_layer >= 1.5) & (do_layer <= 5.0),
            1.0,
            np.clip((do_layer - 1.0) / 0.5, 0, 1) * np.clip((6.0 - do_layer) / 1.0, 0, 1),
        )

        # 綜合: 梯度 × 氧含量適宜性
        edge_enrichment = grad_norm * edge_mask

        log.info(f"  🌊 OMZ edge enrichment: "
                 f"avg={np.nanmean(edge_enrichment):.3f}, "
                 f"max={np.nanmax(edge_enrichment):.3f} "
                 f"(depth_layer={target_depth_idx}, ~{depths[target_depth_idx] if target_depth_idx < len(depths) else '?'}m)")

        return np.clip(edge_enrichment, 0, 1).astype(np.float32)

    def compute_net_omz_effect(
        self,
        species: str,
        compression: "np.ndarray | float",
        edge_enrichment: Optional[np.ndarray] = None,
    ) -> "np.ndarray | float":
        """
        [v12-enhance] 淨 OMZ 效應 — 合併壓縮 + 邊緣富集

        壓縮嚴重但邊緣富集 → 魚可能被擠到邊緣但覓食條件好
        壓縮嚴重且無富集 → 非常差的棲息地

        Parameters:
            species: 物種 ID
            compression: 0-1 壓縮指數
            edge_enrichment: 0-1 邊緣富集

        Returns:
            net_effect: 0-1 淨 OMZ 效應 (加權合併)
        """
        if edge_enrichment is None:
            return compression

        # 深潛物種 (bigeye) 對 OMZ 邊緣更敏感
        edge_weight = {"skipjack": 0.1, "yellowfin": 0.2, "bigeye": 0.4, "albacore": 0.2}
        w_edge = edge_weight.get(species, 0.2)
        w_comp = 1.0 - w_edge

        net = w_comp * compression + w_edge * edge_enrichment

        if isinstance(net, np.ndarray):
            return np.clip(net, 0, 1).astype(np.float32)
        return max(0.0, min(1.0, net))

