"""
OceanMaster v13.2 — DL Models v2
=================================
兩個模型架構 + Physics-informed loss：

【Model A】XGBoost — 立即可用，數據少也能跑
  - 輸入：94維特徵（含 30天 lag + 食物鏈品質特徵）
  - 輸出：HSI分數 + 魚種分類

【Model B】Bi-LSTM + Multi-head Attention — 主力模型
  - 輸入：94維 × 30天時間窗口（Xie 2024: 30天最佳）
  - 輸出：(a) 魚種預測 (b) 漁獲量回歸 (c) 食物鏈 ETA
  - Physics-informed loss + MMH 約束

參考：SEGN (2024), Xie 2024 (U-Net 93.59%), Liu 2024 (zoo RF), Cushing 1990 (MMH)
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("OceanMaster.DLModels")


# ═══════════════════════════════════════════════════════════
#  94 維特徵名定義 (v18.1 擴充)
#  新增 10 維：30天 lag + 浮游動物品質 + 氣候偏移
# ═══════════════════════════════════════════════════════════

FEATURE_NAMES_84 = [
    # 環境基礎 (0-14)
    "sst", "chl", "ssh", "sss", "mld", "z20", "do_surface",
    "wind_speed", "wave_height", "bathy", "npp", "par", "kd490",
    "sst_gradient", "ssh_gradient",
    # 鋋面/渦旋 (15-24)
    "front_intensity", "front_distance_km", "eddy_strength",
    "eddy_type", "eddy_age_days", "kuroshio_distance",
    "u_current", "v_current", "current_shear", "convergence",
    # 食物鏈時序 (25-40) — 擴充到 30天（Xie 2024最佳窗口）
    "chl_lag3", "chl_lag7", "chl_lag14", "chl_lag21", "chl_lag30",
    "npp_lag7", "npp_lag14", "npp_lag21", "npp_lag30",
    "sst_rate_3d", "sst_rate_7d", "sst_rate_14d", "upwelling_index",
    "bloom_active", "bloom_day", "bloom_intensity",
    # 食物鏈品質推估 (41-55) — 新增 5 維
    "bloom_type", "pft_diatom_frac", "pft_micro_frac",
    "zoo_density_est", "baitfish_potential",
    "food_chain_stage", "food_chain_eta",
    "chl_front_intensity", "prey_sst_match",
    "viirs_fishing_light", "npp_change_rate",
    "bloom_age_days",                    # ← MMH (Cushing 1990)
    "diatom_fraction",                   # ← El Hourany 2024
    "copepod_quality_index",             # ← 矽藻×橈腳類尺寸
    "microbial_loop_dominance",          # ← 能量耗散指標
    # 時空/天文 (56-68)
    "day_of_year", "month", "hour_utc", "lunar_phase",
    "solar_elevation", "distance_from_port_nm",
    "distance_to_shelf_nm", "eez_flag",
    "seamount_distance_km", "gfw_fishing_hours", "ais_vessel_density",
    "year_feature",                      # ← 氣候偏移 (Nature Eco.Evo 2023)
    "climate_phenology_offset",          # ← 每年約0.5天
    "zooplankton_biomass_est",           # ← Liu 2024 RF
    # v18.3 氣候指數 ML 特徵 (4 個) — ENSO lag/PDO/SOI
    "enso_lag1",                         # ← 1 年前 ONI (長鰭鮪)
    "enso_lag2",                         # ← 2 年前 ONI (長鰭鮪)
    "pdo_index",                         # ← PDO (黃鰭鮪, Mantua 1997)
    "soi_index",                         # ← SOI (南太平洋長鰭鮪)
    # v18.3 深層溶氧 (3 個) — Liu 2025 ICES / Xu 2025
    "do_50m",                            # ← 50m DO (ml/L)
    "do_150m",                           # ← 150m DO (ml/L)
    "do_200m",                           # ← 200m DO (ml/L)
]

SPECIES_LIST = ["yellowfin", "bigeye", "skipjack", "albacore", "squid"]


# ═══════════════════════════════════════════════════════════
#  Model A: XGBoost Wrapper
# ═══════════════════════════════════════════════════════════

class XGBoostFishingPredictor:
    """
    XGBoost 漁場預測器 — 立即可用，500 筆即可訓練。

    功能：
      1. 漁獲量回歸 (catch_kg)
      2. 魚種分類 (species)
      3. 特徵重要性排序
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.catch_model = None
        self.species_model = None
        self.feature_names = FEATURE_NAMES_84
        self.feature_importance = {}

    def train(
        self,
        X: np.ndarray,
        y_catch: np.ndarray,
        y_species: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        訓練 XGBoost 模型。

        Returns:
            {"catch_rmse", "species_accuracy", "feature_importance"}
        """
        try:
            import xgboost as xgb
        except ImportError:
            log.warning("⚠️ xgboost 未安裝，使用 sklearn GBT 替代")
            return self._train_sklearn(X, y_catch, y_species, sample_weight)

        from sklearn.model_selection import train_test_split

        X_train, X_val, yc_train, yc_val, ys_train, ys_val = train_test_split(
            X, y_catch, y_species, test_size=0.2, random_state=42
        )

        if sample_weight is not None:
            w_train, w_val = train_test_split(sample_weight, test_size=0.2, random_state=42)
        else:
            w_train = w_val = None

        # 漁獲量回歸
        self.catch_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
        )
        self.catch_model.fit(
            X_train, yc_train, sample_weight=w_train,
            eval_set=[(X_val, yc_val)],
            verbose=False,
        )

        # 魚種分類
        n_classes = len(set(y_species))
        self.species_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_class=n_classes if n_classes > 2 else None,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        )
        self.species_model.fit(
            X_train, ys_train, sample_weight=w_train,
            eval_set=[(X_val, ys_val)],
            verbose=False,
        )

        # 特徵重要性
        imp = self.catch_model.feature_importances_
        self.feature_importance = dict(zip(self.feature_names[:len(imp)], imp))
        sorted_feat = sorted(self.feature_importance.items(), key=lambda x: -x[1])

        # 驗證指標
        catch_pred = self.catch_model.predict(X_val)
        catch_rmse = float(np.sqrt(np.mean((catch_pred - yc_val) ** 2)))

        species_pred = self.species_model.predict(X_val)
        species_acc = float(np.mean(species_pred == ys_val))

        results = {
            "catch_rmse": catch_rmse,
            "species_accuracy": species_acc,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "top_10_features": sorted_feat[:10],
        }

        log.info(
            f"✅ XGBoost 訓練完成:\n"
            f"   漁獲 RMSE: {catch_rmse:.2f} kg\n"
            f"   魚種 Acc:  {species_acc:.1%}\n"
            f"   Top 3 特徵: {', '.join(f[0] for f in sorted_feat[:3])}"
        )
        return results

    def _train_sklearn(self, X, y_catch, y_species, sample_weight):
        """sklearn 備援方案。"""
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_val, yc_train, yc_val, ys_train, ys_val = train_test_split(
            X, y_catch, y_species, test_size=0.2, random_state=42
        )

        self.catch_model = GradientBoostingRegressor(n_estimators=200, max_depth=6)
        self.catch_model.fit(X_train, yc_train, sample_weight=sample_weight[:len(X_train)] if sample_weight is not None else None)

        self.species_model = GradientBoostingClassifier(n_estimators=150, max_depth=5)
        self.species_model.fit(X_train, ys_train)

        catch_pred = self.catch_model.predict(X_val)
        catch_rmse = float(np.sqrt(np.mean((catch_pred - yc_val) ** 2)))
        species_acc = float(np.mean(self.species_model.predict(X_val) == ys_val))

        self.feature_importance = dict(zip(
            self.feature_names[:len(self.catch_model.feature_importances_)],
            self.catch_model.feature_importances_
        ))

        return {"catch_rmse": catch_rmse, "species_accuracy": species_acc}

    def predict(self, X: np.ndarray) -> Dict:
        """預測漁獲量和魚種。"""
        result = {}
        if self.catch_model:
            result["catch_kg"] = self.catch_model.predict(X)
        if self.species_model:
            result["species_proba"] = self.species_model.predict_proba(X)
            result["species"] = self.species_model.predict(X)
        return result

    def save(self, tag: str = "latest"):
        """儲存模型 + HMAC-SHA256 簽章。"""
        import pickle
        from engine.ml.stacking_ensemble import _sign_model_file
        path = self.model_dir / f"xgb_fishing_{tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "catch_model": self.catch_model,
                "species_model": self.species_model,
                "feature_importance": self.feature_importance,
                "feature_names": self.feature_names,
            }, f)
        _sign_model_file(path)
        log.info(f"💾 XGBoost saved + signed: {path}")

    def load(self, tag: str = "latest"):
        """載入模型（HMAC-SHA256 驗證後才反序列化）。"""
        import pickle
        from engine.ml.stacking_ensemble import _verify_model_file
        path = self.model_dir / f"xgb_fishing_{tag}.pkl"
        _verify_model_file(path)  # 驗證失敗時 raise RuntimeError
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.catch_model = data["catch_model"]
        self.species_model = data["species_model"]
        self.feature_importance = data.get("feature_importance", {})
        log.info(f"📂 XGBoost loaded (HMAC verified): {path}")


# ═══════════════════════════════════════════════════════════
#  Model B: Bi-LSTM + Multi-Head Attention
# ═══════════════════════════════════════════════════════════

_torch = None


def _ensure_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("PyTorch required. pip install torch")
    return _torch


class MultiHeadAttention:
    """多頭自注意力層（純 PyTorch）。"""

    @staticmethod
    def build(embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        torch = _ensure_torch()
        nn = torch.nn
        return nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)


class BiLSTMAttentionModel:
    """
    Bi-LSTM + Multi-Head Attention 漁場預測模型。

    架構：
      Input (batch, seq_len=30, features=94)
      → Bi-LSTM (2 layers, hidden=128)
      → Multi-Head Self-Attention (4 heads)
      → FC layers
      → 3 輸出頭：
        (a) 魚種分類 (num_species classes)
        (b) 漁獲量回歸 (1 value)
        (c) 食物鏈 ETA (1 value, 預測幾天後到達)

    Physics-informed loss：
      違反生態規則 → 額外懲罰
    """

    @staticmethod
    def build(
        n_features: int = 59,
        seq_len: int = 30,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        n_species: int = 5,
        dropout: float = 0.2,
    ):
        """建構模型。"""
        torch = _ensure_torch()
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 輸入正規化
                self.input_norm = nn.LayerNorm(n_features)

                # Bi-LSTM
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_dim,
                    num_layers=n_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if n_layers > 1 else 0,
                )

                # Multi-Head Attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim * 2,  # Bi-directional
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.attn_norm = nn.LayerNorm(hidden_dim * 2)

                # 共享 backbone
                self.backbone = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )

                # 輸出頭 A: 魚種分類
                self.species_head = nn.Linear(hidden_dim // 2, n_species)

                # 輸出頭 B: 漁獲量回歸
                self.catch_head = nn.Sequential(
                    nn.Linear(hidden_dim // 2, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Softplus(),  # 確保漁獲量為正
                )

                # 輸出頭 C: 食物鏈 ETA (天)
                self.eta_head = nn.Sequential(
                    nn.Linear(hidden_dim // 2, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.ReLU(),  # ETA >= 0
                )

            def forward(self, x):
                """
                Args:
                    x: (batch, seq_len, n_features)
                Returns:
                    species_logits: (batch, n_species)
                    catch_pred: (batch, 1)
                    eta_pred: (batch, 1)
                    attn_weights: (batch, seq_len, seq_len)
                """
                # Layer norm
                x = self.input_norm(x)

                # Bi-LSTM
                lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

                # Self-Attention
                attn_out, attn_weights = self.attention(
                    lstm_out, lstm_out, lstm_out
                )
                attn_out = self.attn_norm(attn_out + lstm_out)  # Residual

                # 取最後一個時間步（或平均）
                context = attn_out[:, -1, :]  # (batch, hidden*2)

                # Backbone
                features = self.backbone(context)

                # 三個輸出頭
                species_logits = self.species_head(features)
                catch_pred = self.catch_head(features)
                eta_pred = self.eta_head(features)

                return species_logits, catch_pred, eta_pred, attn_weights

        model = _Model()
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            f"🧠 Bi-LSTM+Attention 模型建構完成:\n"
            f"   輸入: ({seq_len}, {n_features})\n"
            f"   LSTM: {n_layers}層 × {hidden_dim} (雙向)\n"
            f"   Attention: {n_heads} heads\n"
            f"   輸出: species({n_species}) + catch(1) + eta(1)\n"
            f"   參數量: {n_params:,}"
        )
        return model


# ═══════════════════════════════════════════════════════════
#  Physics-Informed Loss Function
# ═══════════════════════════════════════════════════════════

class PhysicsInformedLoss:
    """
    食物鏈物理約束 Loss。

    違反以下生態規則時增加懲罰：
    1. 黃鰭鮪不出現在 SST < 20°C
    2. 大目鮪需要 Z20 > 150m
    3. bloom_age_days < MMH 餌料魚階段(35天) 不應預測大型鮪魚大量到達
    4. 食物鏈 ETA 不應為負
    5. 矽藻比例低 + 漁獲量高 → 懲罰（微生物環路主導區不提供足夠餌料）
    """

    @staticmethod
    def build(
        penalty_weight: float = 0.1,
        species_col_idx: Dict[str, int] = None,
    ):
        """
        建構 Physics-Informed Loss。

        Returns:
            loss_fn(species_logits, catch_pred, eta_pred,
                    y_species, y_catch, y_eta, x_features)
        """
        torch = _ensure_torch()
        nn = torch.nn

        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        def physics_loss(
            species_logits, catch_pred, eta_pred,
            y_species, y_catch, y_eta, x_features,
        ):
            """
            Combined loss with physics constraints.

            Args:
                species_logits: (batch, n_species)
                catch_pred: (batch, 1)
                eta_pred: (batch, 1)
                y_species: (batch,) long
                y_catch: (batch, 1)
                y_eta: (batch, 1)
                x_features: (batch, seq_len, n_features) — 原始輸入
            """
            # 標準 loss
            loss_species = ce_loss(species_logits, y_species)
            loss_catch = mse_loss(catch_pred, y_catch)
            loss_eta = mse_loss(eta_pred, y_eta)

            # === Physics constraints ===
            penalty = torch.tensor(0.0, device=species_logits.device)

            # 提取最後時間步的環境特徵
            last_step = x_features[:, -1, :]  # (batch, n_features)
            sst = last_step[:, 0]                # idx 0 = sst
            bloom_day = last_step[:, 39]         # idx 39 = bloom_day (updated for 94-dim)
            z20 = last_step[:, 5]                # idx 5 = z20

            species_proba = torch.softmax(species_logits, dim=-1)

            # 規則 1: 黃鰭鮪(idx=0) SST < 20°C → 懲罰
            cold_mask = (sst < 20).float()
            yellowfin_prob = species_proba[:, 0]
            penalty += (cold_mask * yellowfin_prob).mean()

            # 規則 2: 大目鮪(idx=1) Z20 < 150m → 懲罰
            shallow_z20 = (z20 < 150).float()
            bigeye_prob = species_proba[:, 1]
            penalty += (shallow_z20 * bigeye_prob).mean()

            # 規則 3: bloom_day < 10 → 抑制高漁獲預測
            immature_food = (bloom_day < 10).float()
            high_catch = torch.relu(catch_pred.squeeze() - 500)  # 超過 500kg 才懲罰
            penalty += (immature_food * high_catch / 1000).mean()

            # 規則 4: ETA 不為負（已用 ReLU 保證，這裡加額外約束）
            penalty += torch.relu(-eta_pred).mean()

            # ── [v19-causal] 因果鏈驅動的物理約束 ──
            # 基於 Schaefer & Fuller (2010), Prince & Goodyear (2006), Stramma (2012)

            # 規則 5: 鰹魚(idx=2) DO_200m < 3.5 ml/L → 懲罰
            # 鰹魚代謝率最高，對低氧最敏感，3.5 ml/L 為生理極限
            do_200m = last_step[:, 93] if last_step.shape[1] > 93 else torch.full_like(sst, 4.0)
            low_do_skipjack = (do_200m < 3.5).float()
            skipjack_prob = species_proba[:, 2] if species_proba.shape[1] > 2 else torch.zeros_like(sst)
            penalty += 1.5 * (low_do_skipjack * skipjack_prob).mean()

            # 規則 6: 黃鰭鮪(idx=0) DO_200m < 3.0 ml/L → 懲罰
            low_do_yellowfin = (do_200m < 3.0).float()
            penalty += 1.0 * (low_do_yellowfin * yellowfin_prob).mean()

            # 規則 7: 大目鮪(idx=1) DO_200m < 1.5 ml/L → 懲罰
            # 大目鮪有 rete mirabile 系統，耐低氧能力最強
            very_low_do = (do_200m < 1.5).float()
            penalty += 1.0 * (very_low_do * bigeye_prob).mean()

            # 組合 loss
            total = (
                0.4 * loss_species +
                0.3 * loss_catch +
                0.2 * loss_eta +
                penalty_weight * penalty
            )
            return total, {
                "loss_species": loss_species.item(),
                "loss_catch": loss_catch.item(),
                "loss_eta": loss_eta.item(),
                "penalty": penalty.item(),
                "total": total.item(),
            }

        return physics_loss


# ═══════════════════════════════════════════════════════════
#  時間窗口數據集
# ═══════════════════════════════════════════════════════════

class TimeSeriesDataset:
    """將 flat DataFrame 轉為 (batch, seq_len, features) 的時序窗口。"""

    @staticmethod
    def create_windows(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_catch_col: str = "catch_kg",
        target_species_col: str = "species",
        seq_len: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        從 DataFrame 建立滑動窗口。

        對於沒有足夠時序的數據（如單筆紀錄），
        用重複填充到 seq_len。

        Returns:
            X: (n_windows, seq_len, n_features)
            y_catch: (n_windows, 1)
            y_species: (n_windows,) int
            y_eta: (n_windows, 1)
        """
        # 特徵欄位過濾
        available = [c for c in feature_cols if c in df.columns]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            log.warning(f"  ⚠️ 缺少 {len(missing)} 特徵，用 0 填充: {missing[:5]}...")
            for c in missing:
                df[c] = 0.0

        X_flat = df[feature_cols].values.astype(np.float32)
        n_samples, n_features = X_flat.shape

        # 處理 species 標籤
        species_map = {s: i for i, s in enumerate(SPECIES_LIST)}
        y_species_str = df[target_species_col].values
        y_species = np.array([species_map.get(str(s), 0) for s in y_species_str])

        # 漁獲量
        y_catch = df[target_catch_col].values.astype(np.float32).reshape(-1, 1)

        # ETA
        y_eta = df.get("food_chain_eta", pd.Series(np.zeros(n_samples))).values.astype(np.float32).reshape(-1, 1)

        if n_samples >= seq_len:
            # 滑動窗口
            n_windows = n_samples - seq_len + 1
            X_win = np.zeros((n_windows, seq_len, n_features), dtype=np.float32)
            for i in range(n_windows):
                X_win[i] = X_flat[i:i+seq_len]
            y_catch_win = y_catch[seq_len-1:]
            y_species_win = y_species[seq_len-1:]
            y_eta_win = y_eta[seq_len-1:]
        else:
            # 數據不足 → 每筆重複 seq_len 次
            n_windows = n_samples
            X_win = np.zeros((n_windows, seq_len, n_features), dtype=np.float32)
            for i in range(n_windows):
                X_win[i] = np.tile(X_flat[i], (seq_len, 1))
            y_catch_win = y_catch
            y_species_win = y_species
            y_eta_win = y_eta

        log.info(
            f"📦 時序窗口: {n_windows} 個 "
            f"({seq_len} steps × {n_features} features)"
        )
        return X_win, y_catch_win, y_species_win, y_eta_win


# ═══════════════════════════════════════════════════════════
#  CLI 測試
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  DL Models v2 — 架構測試")
    print("=" * 60)

    # 測試 XGBoost
    print("\n--- Model A: XGBoost ---")
    xgb_model = XGBoostFishingPredictor()
    n = 800
    X_test = np.random.randn(n, len(FEATURE_NAMES_84)).astype(np.float32)
    y_catch = np.abs(np.random.randn(n) * 100)
    y_species = np.random.randint(0, 5, n)
    result = xgb_model.train(X_test, y_catch, y_species)
    print(f"  結果: {result}")

    # 測試 Bi-LSTM+Attention
    print("\n--- Model B: Bi-LSTM+Attention ---")
    try:
        model = BiLSTMAttentionModel.build(
            n_features=len(FEATURE_NAMES_84),
            seq_len=30, hidden_dim=64, n_heads=4,
        )
        # 模擬前向傳播
        torch = _ensure_torch()
        x = torch.randn(4, 30, len(FEATURE_NAMES_84))
        sp, catch, eta, attn_w = model(x)
        print(f"  species: {sp.shape}")
        print(f"  catch:   {catch.shape}")
        print(f"  eta:     {eta.shape}")
        print(f"  attn:    {attn_w.shape}")

        # 測試 Physics-Informed Loss
        print("\n--- Physics-Informed Loss ---")
        pi_loss = PhysicsInformedLoss.build(penalty_weight=0.1)
        y_sp = torch.randint(0, 5, (4,))
        y_c = torch.abs(torch.randn(4, 1)) * 100
        y_e = torch.abs(torch.randn(4, 1)) * 10
        total, breakdown = pi_loss(sp, catch, eta, y_sp, y_c, y_e, x)
        print(f"  Loss breakdown: {breakdown}")

    except ImportError as e:
        print(f"  ⚠️ PyTorch 未安裝: {e}")
        print("  Bi-LSTM 不可用，XGBoost 仍可使用")
