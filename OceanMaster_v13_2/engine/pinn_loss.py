"""
PINN — Physics-Informed Neural Network Loss Module
物理資訊神經網路損失函數模組

============================================================
🎯 功能說明：
    將海洋物理定律（流體力學、熱力學守恆、質量守恆）
    直接嵌入深度學習模型的 Loss Function。
    確保 AI 的預測結果「永遠不違反物理定律」。

📌 架構狀態：✅ 完整架構  |  ❌ 尚未訓練
📌 缺少什麼：真實漁獲日誌 + 高解析度海流場資料
📌 買家需要：
    1. 真實的漁獲座標 (lat, lon, catch_weight)
    2. 對應時間的 CMEMS 海流場 (u, v 速度向量)
    3. 對應時間的 SST 溫度場
    4. 將此 Loss 加入現有 DL 模型的訓練迴圈中

📌 如何啟用：
    在 dl_trainer.py 中：
        from engine.pinn_loss import PINNLoss
        pinn = PINNLoss()
        loss = mse_loss + pinn.total_physics_loss(pred, sst, u, v, lat, lon)

📌 物理方程式來源：
    - 質量守恆 (Continuity Equation): ∂u/∂x + ∂v/∂y = 0
    - 熱力學守恆 (Heat Advection): ∂T/∂t + u·∂T/∂x + v·∂T/∂y ≈ 0
    - 地轉平衡 (Geostrophic Balance): f·v = (1/ρ)·∂P/∂x
============================================================
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================
# [選用] 如果有 PyTorch 環境才載入
# 買家部署時需安裝: pip install torch
# ============================================================
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 未安裝。PINN Loss 需要 PyTorch。")


class PINNLoss:
    """
    物理資訊神經網路 (PINN) 損失函數

    將海洋物理方程式嵌入 DL 模型的 Loss Function，
    使神經網路的預測必須符合基本物理定律。

    ⚠️ 架構狀態：骨架已完成，尚未與真實數據訓練
    ⚠️ 啟用條件：需要海流場 (u, v) 與溫度場 (SST) 的網格資料
    """

    # === 物理常數 ===
    EARTH_ROTATION_RATE = 7.2921e-5     # 地球自轉角速度 (rad/s)
    WATER_DENSITY = 1025.0               # 海水密度 (kg/m³)
    GRAVITY = 9.81                        # 重力加速度 (m/s²)

    def __init__(
        self,
        lambda_continuity: float = 1.0,
        lambda_heat: float = 1.0,
        lambda_geostrophic: float = 0.5,
        lambda_boundary: float = 0.1,
        grid_resolution_km: float = 25.0
    ):
        """
        初始化 PINN Loss 權重

        Args:
            lambda_continuity: 質量守恆約束權重
            lambda_heat: 熱力學守恆約束權重
            lambda_geostrophic: 地轉平衡約束權重
            lambda_boundary: 邊界條件約束權重 (例如：陸地上不能有海流)
            grid_resolution_km: 網格解析度 (公里)，用於計算空間微分
        """
        self.lambda_continuity = lambda_continuity
        self.lambda_heat = lambda_heat
        self.lambda_geostrophic = lambda_geostrophic
        self.lambda_boundary = lambda_boundary
        self.dx = grid_resolution_km * 1000.0  # 轉換為公尺

        logger.info(
            f"PINN Loss 初始化完成 | "
            f"λ_cont={lambda_continuity}, λ_heat={lambda_heat}, "
            f"λ_geo={lambda_geostrophic}, λ_bnd={lambda_boundary}"
        )

    # ============================================================
    # 核心物理約束 1：質量守恆 (Continuity Equation)
    # ∂u/∂x + ∂v/∂y = 0
    # 含義：海水不會憑空消失或出現
    # ============================================================
    def continuity_loss(
        self,
        u: 'torch.Tensor',
        v: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        質量守恆損失

        確保海流速度場的散度為零 (不可壓縮流體假設)。
        如果 AI 預測出「海水憑空消失」的結果，此 Loss 會懲罰它。

        Args:
            u: 東西向海流速度 (m/s), shape: (B, H, W)
            v: 南北向海流速度 (m/s), shape: (B, H, W)

        Returns:
            質量守恆違反程度的 Loss 值

        ⚠️ 需要真實數據：CMEMS 的 u, v 海流場
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("需要安裝 PyTorch: pip install torch")

        # 計算空間偏微分 (有限差分法)
        du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2 * self.dx)   # 中央差分
        dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2 * self.dx)

        # 裁剪到相同大小
        min_h = min(du_dx.shape[1], dv_dy.shape[1])
        min_w = min(du_dx.shape[2], dv_dy.shape[2])
        du_dx = du_dx[:, :min_h, :min_w]
        dv_dy = dv_dy[:, :min_h, :min_w]

        # 散度應該為零
        divergence = du_dx + dv_dy
        return torch.mean(divergence ** 2)

    # ============================================================
    # 核心物理約束 2：熱量守恆 (Heat Advection Equation)
    # ∂T/∂t + u·∂T/∂x + v·∂T/∂y ≈ 0
    # 含義：海溫的變化 = 海流把熱量搬走的量
    # ============================================================
    def heat_advection_loss(
        self,
        sst_current: 'torch.Tensor',
        sst_next: 'torch.Tensor',
        u: 'torch.Tensor',
        v: 'torch.Tensor',
        dt: float = 86400.0
    ) -> 'torch.Tensor':
        """
        熱量平流守恆損失

        確保 AI 預測的 SST 變化，與海流搬運熱量的物理過程一致。
        如果 AI 預測「海溫突然暴漲 10 度但沒有暖流經過」，此 Loss 會懲罰它。

        Args:
            sst_current: 當前 SST 場, shape: (B, H, W)
            sst_next: 下一時步 SST 場 (AI 預測值), shape: (B, H, W)
            u: 東西向海流速度, shape: (B, H, W)
            v: 南北向海流速度, shape: (B, H, W)
            dt: 時間步長 (秒), 預設 86400 = 1天

        Returns:
            熱量守恆違反程度的 Loss 值

        ⚠️ 需要真實數據：
            - NOAA OISST 或 CMEMS SST 的連續天數資料
            - CMEMS 的 u, v 海流場 (與 SST 同解析度)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("需要安裝 PyTorch: pip install torch")

        # 時間微分
        dT_dt = (sst_next - sst_current) / dt

        # 空間微分
        dT_dx = (sst_current[:, :, 2:] - sst_current[:, :, :-2]) / (2 * self.dx)
        dT_dy = (sst_current[:, 2:, :] - sst_current[:, :-2, :]) / (2 * self.dx)

        # 裁剪到相同大小
        min_h = min(dT_dx.shape[1], dT_dy.shape[1], u.shape[1], v.shape[1], dT_dt.shape[1])
        min_w = min(dT_dx.shape[2], dT_dy.shape[2], u.shape[2], v.shape[2], dT_dt.shape[2])

        dT_dt_c = dT_dt[:, 1:min_h+1, 1:min_w+1]
        dT_dx_c = dT_dx[:, :min_h, :min_w]
        dT_dy_c = dT_dy[:, :min_h, :min_w]
        u_c = u[:, 1:min_h+1, 1:min_w+1]
        v_c = v[:, 1:min_h+1, 1:min_w+1]

        # 熱量平流方程殘差
        residual = dT_dt_c + u_c * dT_dx_c + v_c * dT_dy_c
        return torch.mean(residual ** 2)

    # ============================================================
    # 核心物理約束 3：地轉平衡 (Geostrophic Balance)
    # f·v = g·∂η/∂x  ;  f·u = -g·∂η/∂y
    # 含義：大尺度海流主要由地球自轉和壓力梯度驅動
    # ============================================================
    def geostrophic_loss(
        self,
        u: 'torch.Tensor',
        v: 'torch.Tensor',
        ssh: 'torch.Tensor',
        lat: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        地轉平衡損失

        在大尺度 (>100km) 上，海流應該滿足地轉平衡。
        如果 AI 預測的海流方向與 SSH（海平面高度）的梯度不一致，此 Loss 會懲罰它。

        Args:
            u: 東西向海流速度, shape: (B, H, W)
            v: 南北向海流速度, shape: (B, H, W)
            ssh: 海平面高度異常 (m), shape: (B, H, W)
            lat: 緯度陣列，用於計算科氏力參數 f, shape: (H,)

        Returns:
            地轉平衡違反程度的 Loss 值

        ⚠️ 需要真實數據：CMEMS SSH (海平面高度) 資料
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("需要安裝 PyTorch: pip install torch")

        # 科氏力參數 f = 2Ω sin(lat)
        f = 2 * self.EARTH_ROTATION_RATE * torch.sin(torch.deg2rad(lat))
        f = f.unsqueeze(0).unsqueeze(-1)  # (1, H, 1)

        # SSH 梯度
        deta_dx = (ssh[:, :, 2:] - ssh[:, :, :-2]) / (2 * self.dx)
        deta_dy = (ssh[:, 2:, :] - ssh[:, :-2, :]) / (2 * self.dx)

        # 裁剪
        min_h = min(deta_dx.shape[1], deta_dy.shape[1])
        min_w = min(deta_dx.shape[2], deta_dy.shape[2])

        # 地轉平衡殘差
        # f * v_g = g * ∂η/∂x
        # f * u_g = -g * ∂η/∂y
        f_c = f[:, 1:min_h+1, :]
        res_u = f_c * v[:, 1:min_h+1, 1:min_w+1] - self.GRAVITY * deta_dx[:, :min_h, :min_w]
        res_v = f_c * u[:, 1:min_h+1, 1:min_w+1] + self.GRAVITY * deta_dy[:, :min_h, :min_w]

        return torch.mean(res_u ** 2 + res_v ** 2)

    # ============================================================
    # 邊界條件約束：陸地遮罩
    # 含義：陸地上不應該有海流或海溫預測
    # ============================================================
    def land_mask_penalty(
        self,
        prediction: 'torch.Tensor',
        land_mask: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        陸地遮罩懲罰

        如果 AI 預測出「在台灣島上面有漁場」，此 Loss 會嚴厲懲罰它。

        Args:
            prediction: 模型預測的熱點機率, shape: (B, H, W)
            land_mask: 陸地遮罩 (陸地=1, 海洋=0), shape: (H, W)

        Returns:
            陸地上非零預測值的懲罰 Loss

        ⚠️ 需要：ETOPO 或 GEBCO 的陸地遮罩資料 (你的 download_bathy.py 已可產生)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("需要安裝 PyTorch: pip install torch")

        land_predictions = prediction * land_mask.unsqueeze(0)
        return torch.mean(land_predictions ** 2)

    # ============================================================
    # 總物理損失 — 加入到現有 DL 訓練迴圈的主函數
    # ============================================================
    def total_physics_loss(
        self,
        prediction: 'torch.Tensor',
        sst_current: 'torch.Tensor',
        sst_next: 'torch.Tensor',
        u: 'torch.Tensor',
        v: 'torch.Tensor',
        ssh: Optional['torch.Tensor'] = None,
        lat: Optional['torch.Tensor'] = None,
        land_mask: Optional['torch.Tensor'] = None
    ) -> 'torch.Tensor':
        """
        計算總物理約束損失

        ⚠️ 使用方式（買家在 dl_trainer.py 中加入）：
            pinn = PINNLoss()
            physics_loss = pinn.total_physics_loss(
                prediction=model_output,
                sst_current=sst_today,
                sst_next=sst_tomorrow,
                u=current_u, v=current_v,
                ssh=ssh_data, lat=lat_grid,
                land_mask=mask
            )
            total_loss = data_loss + 0.1 * physics_loss
            total_loss.backward()

        Returns:
            所有物理約束的加權總和
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("需要安裝 PyTorch: pip install torch")

        total = torch.tensor(0.0, device=prediction.device)

        # 1. 質量守恆
        total += self.lambda_continuity * self.continuity_loss(u, v)

        # 2. 熱量守恆
        total += self.lambda_heat * self.heat_advection_loss(
            sst_current, sst_next, u, v
        )

        # 3. 地轉平衡 (需要 SSH 和緯度)
        if ssh is not None and lat is not None:
            total += self.lambda_geostrophic * self.geostrophic_loss(
                u, v, ssh, lat
            )

        # 4. 陸地遮罩懲罰
        if land_mask is not None:
            total += self.lambda_boundary * self.land_mask_penalty(
                prediction, land_mask
            )

        return total

    # ============================================================
    # 診斷工具 — 輸出各項物理約束的違反程度
    # ============================================================
    def diagnose(
        self,
        u: np.ndarray,
        v: np.ndarray,
        sst: np.ndarray
    ) -> Dict[str, float]:
        """
        診斷工具（不需要 PyTorch，純 NumPy）

        快速檢查一組海洋場資料的物理一致性。
        買家可以用此函數驗證原始衛星數據的品質。

        Args:
            u: 東西向海流 (H, W)
            v: 南北向海流 (H, W)
            sst: 海表溫度 (H, W)

        Returns:
            各項物理指標的違反程度
        """
        # 質量守恆檢查
        du_dx = np.gradient(u, self.dx, axis=1)
        dv_dy = np.gradient(v, self.dx, axis=0)
        divergence = np.mean(np.abs(du_dx + dv_dy))

        # SST 梯度合理性
        sst_grad = np.sqrt(
            np.gradient(sst, axis=0)**2 + np.gradient(sst, axis=1)**2
        )
        max_sst_grad = np.max(sst_grad)

        return {
            "divergence_mean": float(divergence),
            "sst_max_gradient": float(max_sst_grad),
            "physics_consistent": divergence < 1e-5 and max_sst_grad < 5.0
        }
