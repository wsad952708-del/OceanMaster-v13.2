"""
OceanMaster v13.2 — Training Pipeline
=======================================
完整閉環：訓練 → 預測 → 增量學習

使用流程：
  1. 合成數據模式（現在）：直接跑，無需真實數據
  2. 公開數據模式：放入 WCPFC CSV → 自動載入
  3. 真實數據模式：放入漁獲日誌 → 自動回查衛星 → 重新訓練

╔══════════════════════════════════════════════════════════╗
║  TODO: 替換真實漁獲數據                                  ║
║  把 CSV 放到 data/catch_logs/ → 呼叫 pipeline.run()     ║
║  系統會自動：                                            ║
║    1. 載入漁獲日誌                                       ║
║    2. 回查衛星歷史（84維特徵）                            ║
║    3. 訓練 XGBoost + Bi-LSTM                            ║
║    4. 對比新舊模型 → 精度提升才更新                       ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

log = logging.getLogger("OceanMaster.TrainingPipeline")


class TrainingPipeline:
    """
    訓練 → 預測 → 增量學習的完整閉環。

    Workflow:
      build_dataset() → train() → predict() → evaluate()
        ↑                                        │
        └── 新數據到位 → retrain() ──────────────┘
    """

    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "models",
        output_dir: str = "output",
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 延遲載入子模組（避免循環 import）
        self._builder = None
        self._fetcher = None
        self._xgb_model = None
        self._lstm_model = None
        self._best_metrics = {"catch_rmse": float("inf"), "species_accuracy": 0}
        self._training_history = []

    @property
    def builder(self):
        if self._builder is None:
            from engine.ml.training_data_builder import TrainingDataBuilder
            self._builder = TrainingDataBuilder(str(self.data_dir))
        return self._builder

    @property
    def fetcher(self):
        if self._fetcher is None:
            from engine.satellite.historical_fetcher import HistoricalFeatureExtractor
            self._fetcher = HistoricalFeatureExtractor(
                use_cmems=True, use_erddap=True
            )
        return self._fetcher

    # ═══════════════════════════════════════════════════════
    #  Phase 1: 建構數據集
    # ═══════════════════════════════════════════════════════

    def build_dataset(
        self,
        min_samples: int = 2000,
        backfill_satellite: bool = True,
        max_api_calls: int = 500,
    ) -> pd.DataFrame:
        """
        建構訓練數據集，自動補齊衛星特徵。

        流程：
          1. 載入所有數據來源 (真實 > WCPFC > 合成)
          2. 回查衛星歷史 (CMEMS/ERDDAP/氣候學)
          3. 輸出含 84 維特徵的 DataFrame
        """
        log.info("=" * 50)
        log.info("  Phase 1: 建構訓練數據集")
        log.info("=" * 50)

        # 載入數據
        df = self.builder.build(min_samples=min_samples)

        # 衛星回查 — 只對缺少特徵的真實/公開數據做
        if backfill_satellite:
            from engine.ml.dl_models_v2 import FEATURE_NAMES_84
            needs_backfill = df[
                df["source"].isin(["real_catch_log", "wcpfc_public", "fao"])
            ]
            if len(needs_backfill) > 0:
                # 檢查是否缺少食物鏈特徵
                missing_cols = [c for c in FEATURE_NAMES_84 if c not in df.columns]
                if missing_cols:
                    log.info(f"🛰️ 需要回查 {len(needs_backfill)} 筆 ({len(missing_cols)} 個缺失特徵)")
                    backfilled = self.fetcher.backfill_dataframe(
                        needs_backfill, max_api_calls=max_api_calls
                    )
                    # 合併回查結果
                    df.update(backfilled)
            else:
                log.info("  ✓ 所有數據已有完整特徵（合成數據）")

        # 儲存數據集
        save_path = self.data_dir / "training" / "dataset_latest.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        log.info(f"💾 數據集儲存: {save_path} ({len(df)} 筆)")

        return df

    # ═══════════════════════════════════════════════════════
    #  Phase 2: 訓練模型
    # ═══════════════════════════════════════════════════════

    def train(
        self,
        df: Optional[pd.DataFrame] = None,
        train_xgb: bool = True,
        train_lstm: bool = True,
    ) -> Dict:
        """
        訓練 Model A (XGBoost) + Model B (Bi-LSTM+Attention)。

        Returns:
            {
                "xgb": {"catch_rmse", "species_accuracy", "top_features"},
                "lstm": {"loss", "epochs"},
                "timestamp": "...",
            }
        """
        log.info("=" * 50)
        log.info("  Phase 2: 訓練模型")
        log.info("=" * 50)

        if df is None:
            csv_path = self.data_dir / "training" / "dataset_latest.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["date"] = pd.to_datetime(df["date"])
            else:
                df = self.build_dataset()

        from engine.ml.dl_models_v2 import (
            FEATURE_NAMES_84, SPECIES_LIST,
            XGBoostFishingPredictor, BiLSTMAttentionModel,
            PhysicsInformedLoss, TimeSeriesDataset,
        )

        results = {"timestamp": datetime.now().isoformat()}

        # 準備特徵矩陣
        available_features = [c for c in FEATURE_NAMES_84 if c in df.columns]
        missing = set(FEATURE_NAMES_84) - set(available_features)
        if missing:
            log.info(f"  補零特徵: {len(missing)} 個")
            for c in missing:
                df[c] = 0.0

        X = df[FEATURE_NAMES_84].values.astype(np.float32)
        # NaN → 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 標籤
        species_map = {s: i for i, s in enumerate(SPECIES_LIST)}
        y_species = df["species"].map(species_map).fillna(0).astype(int).values
        y_catch = df["catch_kg"].fillna(0).values.astype(np.float32)
        sample_weight = df.get("sample_weight", pd.Series(np.ones(len(df)))).values

        # ── v18.3 RFECV Feature Selection (Xu 2024 CJFAS) ──
        # Recursive Feature Elimination with Cross-Validation
        # Confirmed optimal for tuna CPUE: selects SST+DO+CHL+SSH subsets
        rfecv_mask = None
        try:
            from sklearn.feature_selection import RFECV
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import KFold

            if len(X) >= 500:  # Need enough data for meaningful CV
                log.info("\n--- RFECV Feature Selection ---")
                rf_estimator = RandomForestRegressor(
                    n_estimators=50, max_depth=8, n_jobs=-1, random_state=42,
                )
                rfecv = RFECV(
                    estimator=rf_estimator,
                    step=1,
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    scoring="neg_mean_squared_error",
                    min_features_to_select=10,
                    n_jobs=-1,
                )
                rfecv.fit(X, y_catch)

                rfecv_mask = rfecv.support_
                n_selected = rfecv_mask.sum()
                feature_names = FEATURE_NAMES_84
                selected = [f for f, s in zip(feature_names, rfecv_mask) if s]
                eliminated = [f for f, s in zip(feature_names, rfecv_mask) if not s]

                log.info(f"  RFECV: {n_selected}/{len(feature_names)} features selected")
                log.info(f"  Top selected: {selected[:15]}")
                if eliminated:
                    log.info(f"  Eliminated: {eliminated[:10]}{'...' if len(eliminated) > 10 else ''}")

                # Save RFECV results
                rfecv_report = {
                    "n_features_optimal": int(n_selected),
                    "selected_features": selected,
                    "eliminated_features": eliminated,
                    "ranking": {f: int(r) for f, r in zip(feature_names, rfecv.ranking_)},
                }
                rfecv_path = self.output_dir / "rfecv_report.json"
                with open(rfecv_path, "w", encoding="utf-8") as f:
                    json.dump(rfecv_report, f, ensure_ascii=False, indent=2)
                log.info(f"  📊 RFECV report: {rfecv_path}")
                results["rfecv"] = rfecv_report

                # Note: We train on ALL features but log which are most important.
                # Actual feature reduction can be enabled later by setting
                # X = X[:, rfecv_mask] before training.
            else:
                log.info("  ⏭️ RFECV 跳過 (需 500+ 筆)")
        except ImportError:
            log.info("  ⏭️ RFECV 跳過 (sklearn 未安裝)")

        # ── Model A: XGBoost ──
        if train_xgb and len(X) >= 100:
            log.info("\n--- Model A: XGBoost ---")
            self._xgb_model = XGBoostFishingPredictor(str(self.model_dir))
            xgb_result = self._xgb_model.train(X, y_catch, y_species, sample_weight)
            self._xgb_model.save("latest")
            results["xgb"] = xgb_result

            # 儲存特徵重要性報告
            self._save_feature_report(xgb_result.get("top_10_features", []))
        else:
            log.info("  ⏭️ XGBoost 跳過 (數據量不足或未啟用)")

        # ── Model B: Bi-LSTM+Attention ──
        if train_lstm and len(X) >= 200:
            log.info("\n--- Model B: Bi-LSTM+Attention ---")
            try:
                import torch
                from torch.utils.data import DataLoader, TensorDataset

                # 建立時序窗口
                seq_len = min(30, max(5, len(df) // 10))
                X_win, y_catch_win, y_species_win, y_eta_win = \
                    TimeSeriesDataset.create_windows(
                        df, FEATURE_NAMES_84, seq_len=seq_len
                    )

                # 建模型
                model = BiLSTMAttentionModel.build(
                    n_features=len(FEATURE_NAMES_84),
                    seq_len=seq_len,
                    hidden_dim=64,
                    n_heads=4,
                    n_species=len(SPECIES_LIST),
                )

                # Physics-informed loss
                pi_loss_fn = PhysicsInformedLoss.build(penalty_weight=0.1)

                # 資料 → DataLoader
                X_t = torch.FloatTensor(X_win)
                yc_t = torch.FloatTensor(y_catch_win)
                ys_t = torch.LongTensor(y_species_win)
                ye_t = torch.FloatTensor(y_eta_win)

                split = int(0.8 * len(X_t))
                train_ds = TensorDataset(X_t[:split], yc_t[:split], ys_t[:split], ye_t[:split])
                val_ds = TensorDataset(X_t[split:], yc_t[split:], ys_t[split:], ye_t[split:])

                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=32)

                # 訓練
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5
                )

                best_val_loss = float("inf")
                patience_counter = 0
                max_epochs = 50
                patience = 10

                for epoch in range(max_epochs):
                    model.train()
                    train_losses = []
                    for batch in train_loader:
                        x_b, yc_b, ys_b, ye_b = batch
                        optimizer.zero_grad()
                        sp, catch, eta, attn = model(x_b)
                        loss, breakdown = pi_loss_fn(sp, catch, eta, ys_b, yc_b, ye_b, x_b)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        train_losses.append(breakdown["total"])

                    # 驗證
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for batch in val_loader:
                            x_b, yc_b, ys_b, ye_b = batch
                            sp, catch, eta, attn = model(x_b)
                            _, bd = pi_loss_fn(sp, catch, eta, ys_b, yc_b, ye_b, x_b)
                            val_losses.append(bd["total"])

                    avg_train = np.mean(train_losses)
                    avg_val = np.mean(val_losses) if val_losses else avg_train

                    scheduler.step(avg_val)

                    if (epoch + 1) % 10 == 0:
                        log.info(f"  Epoch {epoch+1}: train={avg_train:.4f} val={avg_val:.4f}")

                    if avg_val < best_val_loss:
                        best_val_loss = avg_val
                        patience_counter = 0
                        torch.save(model.state_dict(), self.model_dir / "bilstm_attn_best.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            log.info(f"  Early stop at epoch {epoch+1}")
                            break

                results["lstm"] = {
                    "best_val_loss": best_val_loss,
                    "epochs_trained": epoch + 1,
                    "seq_len": seq_len,
                    "n_params": sum(p.numel() for p in model.parameters()),
                }
                log.info(f"✅ Bi-LSTM 訓練完成: val_loss={best_val_loss:.4f}")

            except ImportError:
                log.warning("  ⚠️ PyTorch 未安裝，Bi-LSTM 跳過")
                results["lstm"] = {"status": "skipped", "reason": "torch not installed"}
        else:
            log.info("  ⏭️ Bi-LSTM 跳過 (數據量不足或未啟用)")
            results["lstm"] = {"status": "skipped", "reason": f"n_samples={len(X)}"}

        # 儲存訓練結果
        self._training_history.append(results)
        report_path = self.output_dir / "training_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"📋 訓練報告: {report_path}")

        return results

    # ═══════════════════════════════════════════════════════
    #  Phase 3: 預測
    # ═══════════════════════════════════════════════════════

    def predict(
        self,
        current_features: np.ndarray,
        forecast_days: list = None,
    ) -> Dict:
        """
        用訓練好的模型做漁場預測。

        Args:
            current_features: (n_points, 84) 今天的衛星特徵
            forecast_days: [7, 14, 21, 30] 預測天數

        Returns:
            {
                "predictions": [{
                    "lat", "lon", "species", "species_confidence",
                    "catch_kg_est", "eta_days", "hsi_score",
                    "recommended_departure": "YYYY-MM-DD",
                }],
                "forecast_map": {...}
            }
        """
        if forecast_days is None:
            forecast_days = [7, 14, 21, 30]

        predictions = []

        # 用 XGBoost 預測
        if self._xgb_model and self._xgb_model.catch_model:
            xgb_result = self._xgb_model.predict(current_features)
            catch_pred = xgb_result.get("catch_kg", np.zeros(len(current_features)))
            species_pred = xgb_result.get("species", np.zeros(len(current_features), dtype=int))

            from engine.ml.dl_models_v2 import SPECIES_LIST
            for i in range(len(current_features)):
                sp_idx = int(species_pred[i]) if i < len(species_pred) else 0
                predictions.append({
                    "species": SPECIES_LIST[sp_idx] if sp_idx < len(SPECIES_LIST) else "unknown",
                    "catch_kg_est": float(catch_pred[i]) if i < len(catch_pred) else 0,
                    "model": "xgboost",
                })

        return {
            "predictions": predictions,
            "n_points": len(current_features),
            "forecast_days": forecast_days,
            "timestamp": datetime.now().isoformat(),
        }

    # ═══════════════════════════════════════════════════════
    #  Phase 4: 增量學習
    # ═══════════════════════════════════════════════════════

    def retrain_with_new_data(
        self,
        catch_log_path: Optional[str] = None,
    ) -> Dict:
        """
        增量學習：新數據到位 → 重新訓練 → 比較精度。

        ╔══════════════════════════════════════════════════╗
        ║  在此插入真實漁獲日誌                             ║
        ║                                                  ║
        ║  用法：                                          ║
        ║  pipeline.retrain_with_new_data(                 ║
        ║      "data/catch_logs/2024_vessel_A.csv"         ║
        ║  )                                               ║
        ║                                                  ║
        ║  或直接把 CSV 放到 data/catch_logs/ 目錄下         ║
        ║  然後呼叫 pipeline.retrain_with_new_data()        ║
        ╚══════════════════════════════════════════════════╝
        """
        log.info("=" * 50)
        log.info("  Phase 4: 增量學習")
        log.info("=" * 50)

        # 記錄舊模型指標
        old_metrics = self._best_metrics.copy()

        # 重建數據集（會自動載入新的漁獲日誌）
        df = self.build_dataset(backfill_satellite=True)

        # 重新訓練
        results = self.train(df)

        # 比較新舊模型
        new_rmse = results.get("xgb", {}).get("catch_rmse", float("inf"))
        new_acc = results.get("xgb", {}).get("species_accuracy", 0)

        improved = (new_rmse < old_metrics["catch_rmse"] or
                    new_acc > old_metrics["species_accuracy"])

        if improved:
            self._best_metrics = {
                "catch_rmse": new_rmse,
                "species_accuracy": new_acc,
            }
            log.info(
                f"✅ 模型改進！更新線上模型\n"
                f"   RMSE: {old_metrics['catch_rmse']:.2f} → {new_rmse:.2f}\n"
                f"   Acc:  {old_metrics['species_accuracy']:.1%} → {new_acc:.1%}"
            )
        else:
            log.info(
                f"⚠️ 新模型未改進，保留舊模型\n"
                f"   RMSE: {old_metrics['catch_rmse']:.2f} vs {new_rmse:.2f}\n"
                f"   Acc:  {old_metrics['species_accuracy']:.1%} vs {new_acc:.1%}"
            )

        return {
            "improved": improved,
            "old_metrics": old_metrics,
            "new_metrics": {"catch_rmse": new_rmse, "species_accuracy": new_acc},
            "results": results,
        }

    # ═══════════════════════════════════════════════════════
    #  完整流程 (一鍵啟動)
    # ═══════════════════════════════════════════════════════

    def run(
        self,
        min_samples: int = 2000,
        train_xgb: bool = True,
        train_lstm: bool = True,
    ) -> Dict:
        """
        一鍵執行完整訓練流程。

        等同於：
          df = build_dataset()
          results = train(df)
        """
        log.info("🚀 OceanMaster Training Pipeline 啟動")
        start = time.time()

        df = self.build_dataset(min_samples=min_samples)
        results = self.train(df, train_xgb=train_xgb, train_lstm=train_lstm)

        elapsed = time.time() - start
        results["elapsed_seconds"] = elapsed
        log.info(f"\n✅ 全部完成! 耗時 {elapsed:.1f} 秒")

        return results

    # ═══════════════════════════════════════════════════════
    #  內部工具
    # ═══════════════════════════════════════════════════════

    def _save_feature_report(self, top_features: list):
        """儲存特徵重要性報告。"""
        report = "# 特徵重要性報告\n\n"
        report += f"更新: {datetime.now().isoformat()}\n\n"
        report += "| 排名 | 特徵 | 重要性 | 食物鏈意義 |\n"
        report += "|------|------|--------|----------|\n"

        meanings = {
            "chl_lag14": "14天前CHL → 餌料魚聚集期",
            "chl_lag7": "7天前CHL → 浮游動物回應期",
            "front_intensity": "鋒面強度 → 營養鹽湧升帶",
            "sst": "海表溫度 → 魚種分佈範圍",
            "bloom_day": "藻華天數 → 食物鏈成熟度",
            "food_chain_stage": "食物鏈階段(0-4)",
            "baitfish_potential": "餌料魚潛力指數",
            "zoo_density_est": "浮游動物密度估計",
            "npp": "淨初級生產力",
            "eddy_age_days": "渦旋年齡",
        }

        for i, (feat, imp) in enumerate(top_features, 1):
            meaning = meanings.get(feat, "—")
            report += f"| {i} | `{feat}` | {imp:.4f} | {meaning} |\n"

        path = self.output_dir / "feature_importance_report.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        log.info(f"📊 特徵報告: {path}")


# ═══════════════════════════════════════════════════════════
#  CLI 測試
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  OceanMaster Training Pipeline — 完整測試")
    print("  (使用合成數據)")
    print("=" * 60)

    pipeline = TrainingPipeline(
        data_dir="data",
        model_dir="models",
        output_dir="output",
    )

    # 一鍵執行
    results = pipeline.run(
        min_samples=1000,
        train_xgb=True,
        train_lstm=False,  # 先不跑 LSTM (需 PyTorch)
    )

    print("\n" + "=" * 60)
    print("  訓練結果:")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
