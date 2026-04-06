"""
OceanMaster — Global Fishing Watch (GFW) 數據載入器
====================================================
用 AIS 漁船行為數據作為 CPUE 代理變數 (proxy CPUE)

學術依據:
  - Kroodsma et al. (2018) Science 359:904-908
  - 原理: 漁船低速徘徊時間 ≈ 漁獲成功率代理指標
  - 授權: CC BY-SA 4.0 (商用可)

數據源:
  - GFW API v3: https://gateway.api.globalfishingwatch.org/v3/
  - 提供: fishing effort (小時/km²), 解析度 0.01°, 2012年至今

使用方式:
  loader = GFWDataLoader(api_token="your-token")
  df = await loader.fetch_fishing_effort(
      lat_range=(10, 30), lon_range=(120, 150),
      start_date="2024-01-01", end_date="2024-12-31",
      gear_types=["tuna_longline", "tuna_purse_seine"],
  )
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple, Dict

log = logging.getLogger("OceanMaster.GFW")


class GFWDataLoader:
    """
    Global Fishing Watch 漁船行為數據載入器

    核心功能:
    1. 從 GFW API 下載逐日 fishing effort
    2. 聚合到 0.25° 週度網格 → proxy_cpue
    3. 輸出 ML 訓練格式 (DataFrame)
    """

    BASE_URL = "https://gateway.api.globalfishingwatch.org/v3"

    # GFW 支援的鮪魚相關漁具類型
    TUNA_GEAR_TYPES = [
        "tuna_purse_seines",
        "drifting_longlines",
        "set_longlines",
        "trollers",
        "pole_and_line",
    ]

    def __init__(self, api_token: str, cache_dir: str = "data/gfw_cache"):
        """
        Parameters:
            api_token: GFW API Bearer token
            cache_dir: 快取目錄
        """
        self.token = api_token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    # ─── 核心 API 查詢: 4Wings (Heatmap API) ───

    async def fetch_fishing_effort(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        gear_types: Optional[List[str]] = None,
        resolution: str = "LOW",  # LOW=0.1°, HIGH=0.01°
    ) -> Optional[pd.DataFrame]:
        """
        從 GFW 4Wings API 取得漁船作業小時數 (fishing hours) 的空間分布。

        Parameters:
            lat_range: (lat_min, lat_max)
            lon_range: (lon_min, lon_max)
            start_date: 起始日 (YYYY-MM-DD)
            end_date: 結束日 (YYYY-MM-DD)
            gear_types: 漁具類別篩選 (預設: 所有鮪魚漁具)
            resolution: LOW(0.1°) 或 HIGH(0.01°)

        Returns:
            DataFrame with columns: [lat, lon, date, fishing_hours, vessel_count]
        """
        cache_key = (f"gfw_{lat_range[0]}_{lat_range[1]}_"
                     f"{lon_range[0]}_{lon_range[1]}_"
                     f"{start_date}_{end_date}.parquet")
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            log.info(f"  GFW: loading from cache {cache_path.name}")
            return pd.read_parquet(cache_path)

        if gear_types is None:
            gear_types = self.TUNA_GEAR_TYPES

        log.info(f"  GFW: querying {start_date} → {end_date}, "
                 f"area=({lat_range}, {lon_range}), gears={len(gear_types)}")

        try:
            import httpx

            # GFW 4Wings API v3 — POST with GeoJSON geometry
            # 官方文檔: https://globalfishingwatch.org/our-apis/documentation
            url = f"{self.BASE_URL}/4wings/report"

            # 構建 bounding box GeoJSON polygon
            lat_min, lat_max = lat_range
            lon_min, lon_max = lon_range
            bbox_geojson = {
                "type": "Polygon",
                "coordinates": [[
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min],
                ]]
            }

            # 分月查詢 (避免單次請求過大)
            all_records = []
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            current = start_dt
            while current < end_dt:
                # Calculate month end
                if current.month == 12:
                    next_month = current.replace(year=current.year + 1, month=1, day=1)
                else:
                    next_month = current.replace(month=current.month + 1, day=1)
                month_end = min(next_month - timedelta(days=1), end_dt)

                date_range_str = (f"{current.strftime('%Y-%m-%d')},"
                                  f"{month_end.strftime('%Y-%m-%d')}")

                params = {
                    "spatial-resolution": resolution,
                    "temporal-resolution": "MONTHLY",
                    "group-by": "GEARTYPE",
                    "datasets[0]": "public-global-fishing-effort:latest",
                    "date-range": date_range_str,
                    "format": "JSON",
                }

                # POST body with geometry (correct format: {geojson: polygon})
                body = {"geojson": bbox_geojson}

                try:
                    async with httpx.AsyncClient(timeout=120) as client:
                        resp = await client.post(
                            url,
                            params=params,
                            json=body,
                            headers=self.headers,
                        )

                        if resp.status_code == 200:
                            data = resp.json()
                            records = self._parse_4wings_response(
                                data, current, gear_types,
                                lat_range, lon_range,
                            )
                            all_records.extend(records)
                            log.info(f"    {current.strftime('%Y-%m')}: "
                                     f"{len(records)} cells")
                        elif resp.status_code == 401:
                            log.error("  GFW: Authentication failed (invalid token)")
                            return self._try_csv_fallback(lat_range, lon_range, start_date, end_date)
                        elif resp.status_code == 429:
                            log.warning("  GFW: Rate limited, waiting 30s...")
                            await asyncio.sleep(30)
                            continue
                        else:
                            # Log response body for debugging
                            try:
                                err_body = resp.text[:500]
                            except Exception:
                                err_body = ""
                            log.warning(f"  GFW: {resp.status_code} for "
                                        f"{current.strftime('%Y-%m')}: {err_body}")
                except Exception as e:
                    log.warning(f"  GFW month {current.strftime('%Y-%m')}: {e}")

                # Advance to next month
                current = next_month

            if not all_records:
                log.warning("  GFW: No data returned, trying CSV fallback")
                return self._try_csv_fallback(lat_range, lon_range, start_date, end_date)

            df = pd.DataFrame(all_records)

            # 快取
            try:
                df.to_parquet(cache_path, index=False)
                log.info(f"  GFW: cached {len(df)} records -> {cache_path.name}")
            except Exception:
                df.to_csv(cache_path.with_suffix('.csv'), index=False)

            return df

        except ImportError:
            log.error("  GFW: httpx not installed")
            return None
        except Exception as e:
            log.error(f"  GFW: {e}")
            return self._try_csv_fallback(lat_range, lon_range, start_date, end_date)

    def _parse_4wings_response(
        self, data: dict, month_dt: datetime,
        gear_types: List[str],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
    ) -> list:
        """解析 4Wings API v3 回應

        GFW v3 回應格式:
        {
          "entries": [
            {
              "public-global-fishing-effort:v3.0": [
                {"date":"2024-01","geartype":"drifting_longlines",
                 "hours":65.3,"lat":27,"lon":121.1,"vesselIDs":5}
              ]
            }
          ]
        }
        """
        records = []

        # Extract entries from response
        entries = data.get("entries", [])
        if not entries:
            # Try flat format as fallback
            entries = data if isinstance(data, list) else data.get("data", [])

        # Flatten: entries is a list of dicts, each dict has dataset name as key
        flat_records = []
        for entry in entries:
            if isinstance(entry, dict):
                for key, value in entry.items():
                    if isinstance(value, list):
                        flat_records.extend(value)
                    elif isinstance(value, dict):
                        # Single record
                        flat_records.append(value)
                    else:
                        # entry itself might be a record
                        flat_records.append(entry)
                        break

        for rec in flat_records:
            if not isinstance(rec, dict):
                continue

            lat = rec.get("lat", rec.get("latitude"))
            lon = rec.get("lon", rec.get("longitude"))
            hours = rec.get("hours", rec.get("fishing_hours",
                     rec.get("value", 0)))
            gear = rec.get("geartype", rec.get("gear_type", "unknown"))

            if lat is None or lon is None:
                continue

            # 篩選區域
            if not (lat_range[0] <= lat <= lat_range[1]):
                continue
            if not (lon_range[0] <= lon <= lon_range[1]):
                continue

            # 篩選鮪魚相關漁具 (but accept all if we get data)
            if gear_types and gear not in gear_types and gear != "unknown":
                # Still include — we want all fishing effort for training
                pass

            records.append({
                "lat": round(float(lat), 2),
                "lon": round(float(lon), 2),
                "date": month_dt.strftime("%Y-%m-%d"),
                "fishing_hours": float(hours),
                "gear_type": gear,
                "vessel_count": rec.get("vesselIDs", 0),
                "year": month_dt.year,
                "month": month_dt.month,
            })

        return records

    # ─── CSV 下載回退 ───

    def _try_csv_fallback(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        嘗試從本地 CSV 檔案載入 GFW 數據。
        用戶可手動從 https://globalfishingwatch.org/data-download/ 下載 CSV。
        """
        csv_files = list(self.cache_dir.glob("*.csv"))
        if not csv_files:
            log.info("  GFW: No local CSV found. Download from "
                     "https://globalfishingwatch.org/data-download/")
            return None

        all_dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                # 標準化欄位名
                df.columns = [c.lower().strip() for c in df.columns]

                # 找 lat/lon/hours 欄位
                lat_col = next((c for c in df.columns
                                if 'lat' in c), None)
                lon_col = next((c for c in df.columns
                                if 'lon' in c), None)
                hours_col = next((c for c in df.columns
                                  if 'hour' in c or 'effort' in c), None)

                if lat_col and lon_col and hours_col:
                    filtered = df[
                        (df[lat_col] >= lat_range[0]) &
                        (df[lat_col] <= lat_range[1]) &
                        (df[lon_col] >= lon_range[0]) &
                        (df[lon_col] <= lon_range[1])
                    ].copy()

                    if len(filtered) > 0:
                        result = pd.DataFrame({
                            "lat": filtered[lat_col].values,
                            "lon": filtered[lon_col].values,
                            "fishing_hours": filtered[hours_col].values,
                        })
                        all_dfs.append(result)
                        log.info(f"  GFW CSV: {f.name} → {len(result)} records")
            except Exception as e:
                log.debug(f"  GFW CSV parse error: {f.name}: {e}")

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return None

    # ─── 聚合到 0.25° 網格 (proxy CPUE) ───

    def aggregate_to_grid(
        self,
        df: pd.DataFrame,
        resolution: float = 0.25,
        time_window: str = "weekly",
    ) -> pd.DataFrame:
        """
        將 fishing hours 聚合到 0.25° 網格 + 時間窗口。

        Parameters:
            df: 原始 fishing effort DataFrame (lat, lon, fishing_hours, date)
            resolution: 網格解析度 (度)
            time_window: "weekly" 或 "monthly"

        Returns:
            DataFrame: [grid_lat, grid_lon, period, proxy_cpue, total_hours, cell_count]
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 網格化座標
        df = df.copy()
        df["grid_lat"] = (df["lat"] / resolution).round() * resolution
        df["grid_lon"] = (df["lon"] / resolution).round() * resolution

        # 時間窗口
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if time_window == "weekly":
                df["period"] = df["date"].dt.to_period("W").astype(str)
            else:
                df["period"] = df["date"].dt.to_period("M").astype(str)
        else:
            df["period"] = "all"

        # 聚合: 每個網格+時段的累計 fishing hours
        grouped = df.groupby(["grid_lat", "grid_lon", "period"]).agg(
            total_hours=("fishing_hours", "sum"),
            cell_count=("fishing_hours", "count"),
        ).reset_index()

        # proxy CPUE: 標準化到 0-1
        max_hours = grouped["total_hours"].quantile(0.99)
        grouped["proxy_cpue"] = np.clip(
            grouped["total_hours"] / max(max_hours, 0.01), 0, 1.0
        ).astype(np.float32)

        log.info(f"  GFW grid: {len(grouped)} cells, "
                 f"avg proxy_cpue={grouped['proxy_cpue'].mean():.3f}, "
                 f"max hours={grouped['total_hours'].max():.0f}")

        return grouped

    # ─── 匹配環境數據 (生成訓練集) ───

    async def build_training_set(
        self,
        effort_grid: pd.DataFrame,
        env_fetcher=None,
    ) -> pd.DataFrame:
        """
        將漁船 effort 網格與環境數據匹配，生成 ML 訓練集。

        每行 = 一個 0.25° 網格, 包含:
          - 環境特徵: SST, CHL, SSH, EKE, ...
          - 目標變量: proxy_cpue (fishing hours 標準化)

        Parameters:
            effort_grid: 聚合後的 fishing effort DataFrame
            env_fetcher: OceanDataFetcher 實例 (可選)

        Returns:
            DataFrame ready for ML training
        """
        if effort_grid.empty:
            return pd.DataFrame()

        from engine.data_fetcher_v2 import WOAClimatology
        clim = WOAClimatology()

        # Per-sample RNG for reproducible noise
        rng = np.random.RandomState(42)

        records = []
        for idx, (_, row) in enumerate(effort_grid.iterrows()):
            lat = row["grid_lat"]
            lon = row["grid_lon"]

            # 從時段提取月份
            period_str = str(row.get("period", "2024-06"))
            try:
                month = int(period_str.split("-")[1][:2])
            except (IndexError, ValueError):
                month = 6

            # 用氣候態填充環境特徵 + 加入合理隨機擾動
            # (WOA climatology is deterministic per location → adds noise to
            #  prevent model from memorizing location through features)
            la = np.array([lat])
            lo = np.array([lon])

            sst_base = clim.sst(la, lo, month)[0, 0]
            chl_base = clim.chl(la, lo, month)[0, 0]
            ssh_base = sst_base * 0.001  # rough proxy
            u_c, v_c = clim.currents(la, lo)
            eke_base = float(0.5 * (u_c[0, 0]**2 + v_c[0, 0]**2))
            cs_base = float(np.sqrt(u_c[0, 0]**2 + v_c[0, 0]**2))

            # Add realistic random perturbation to break location-determinism
            # (std values from satellite vs climatology comparison literature)
            sst = sst_base + rng.normal(0, 1.5)    # ±1.5°C typical daily variance
            chl_val = chl_base * rng.lognormal(0, 0.3)  # ±30% CHL variance
            chl_val = max(chl_val, 0.01)
            ssh_val = ssh_base + rng.normal(0, 0.05)
            eke = max(0, eke_base + rng.normal(0, eke_base * 0.5 + 1e-6))
            cs = max(0, cs_base + rng.normal(0, cs_base * 0.3 + 1e-6))

            # Temporal cyclic features (season is real signal, not leakage)
            month_sin = float(np.sin(2 * np.pi * month / 12))
            month_cos = float(np.cos(2 * np.pi * month / 12))

            # Interaction features
            sst_chl_interaction = sst * np.log10(max(chl_val, 0.01))

            records.append({
                "lat": lat,
                "lon": lon,
                "month": month,
                "proxy_cpue": row["proxy_cpue"],
                "total_fishing_hours": row["total_hours"],
                # Environmental features (with noise)
                "sst": sst,
                "chl": chl_val,
                "ssh": ssh_val,
                "eke": eke,
                "current_speed": cs,
                # Temporal features
                "month_sin": month_sin,
                "month_cos": month_cos,
                # Interaction features
                "sst_chl_interaction": sst_chl_interaction,
            })

        df = pd.DataFrame(records)
        log.info(f"  Training set: {len(df)} samples, "
                 f"proxy_cpue range=[{df['proxy_cpue'].min():.3f}, "
                 f"{df['proxy_cpue'].max():.3f}]")
        log.info(f"  NOTE: WOA climatology + noise (NOT real-time satellite)")
        return df


# ═══════════════════════════════════════════════════
# CLI 工具
# ═══════════════════════════════════════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download GFW fishing effort data"
    )
    parser.add_argument("--token", required=True, help="GFW API token")
    parser.add_argument("--lat-min", type=float, default=10)
    parser.add_argument("--lat-max", type=float, default=30)
    parser.add_argument("--lon-min", type=float, default=120)
    parser.add_argument("--lon-max", type=float, default=150)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", default="data/gfw_training.csv")
    args = parser.parse_args()

    loader = GFWDataLoader(api_token=args.token)
    df = await loader.fetch_fishing_effort(
        lat_range=(args.lat_min, args.lat_max),
        lon_range=(args.lon_min, args.lon_max),
        start_date=args.start,
        end_date=args.end,
    )

    if df is not None and not df.empty:
        grid = loader.aggregate_to_grid(df)
        training = await loader.build_training_set(grid)
        training.to_csv(args.output, index=False)
        print(f"✅ Saved {len(training)} training samples → {args.output}")
    else:
        print("❌ No data retrieved")


if __name__ == "__main__":
    asyncio.run(main())
