"""
OceanMaster v10.5 — Low-Bandwidth Data Packer
================================================
極低頻寬封裝引擎: VSAT 衛星網路 ≤500KB payload。

封裝策略:
  - 向量 (hotspots/fronts): GeoJSON → msgpack/zlib
  - 網格 (SST/EKE): float32 → int16 quantize → zlib
  - 元資料: JSON minimal
"""
import io
import json
import zlib
import struct
import logging
import numpy as np
from typing import Any, Dict, List, Optional

log = logging.getLogger("OceanMaster.packer")

# ── Payload budget ──
MAX_PAYLOAD_BYTES = 500 * 1024  # 500 KB hard limit


def pack_daily_forecast(
    hotspots: List[Dict],
    sst_grid: Optional[np.ndarray] = None,
    eke_grid: Optional[np.ndarray] = None,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
    compression_level: int = 9,
) -> bytes:
    """
    打包每日預報為極限壓縮二進位格式。

    Format (v1):
      [4B magic] [4B version] [4B flags]
      [4B meta_len] [meta_zlib]
      [4B hotspot_len] [hotspot_zlib]
      [4B grid_count] [grid_header+data]*

    Returns:
        bytes — 壓縮封包, ≤500KB
    """
    buf = io.BytesIO()

    # ── Header ──
    buf.write(b"OMPK")  # magic
    buf.write(struct.pack("<I", 1))  # version
    flags = 0
    if sst_grid is not None:
        flags |= 0x01
    if eke_grid is not None:
        flags |= 0x02
    buf.write(struct.pack("<I", flags))

    # ── Metadata (JSON → zlib) ──
    meta = metadata or {}
    meta_slim = {
        "version": meta.get("version", "10.5"),
        "timestamp": meta.get("timestamp", ""),
        "n_hotspots": len(hotspots),
        "grid_shape": [int(x) for x in (sst_grid.shape if sst_grid is not None else [0, 0])],
    }
    if lats is not None:
        meta_slim["lat_range"] = [round(float(lats.min()), 2), round(float(lats.max()), 2)]
        meta_slim["lon_range"] = [round(float(lons.min()), 2), round(float(lons.max()), 2)]
    meta_bytes = zlib.compress(json.dumps(meta_slim).encode("utf-8"), compression_level)
    buf.write(struct.pack("<I", len(meta_bytes)))
    buf.write(meta_bytes)

    # ── Hotspots (slim JSON → zlib) ──
    slim_hotspots = _slim_hotspots(hotspots)
    hs_json = json.dumps(slim_hotspots, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    hs_compressed = zlib.compress(hs_json, compression_level)
    buf.write(struct.pack("<I", len(hs_compressed)))
    buf.write(hs_compressed)

    # ── Grid data (quantize int16 → zlib) ──
    grids = []
    if sst_grid is not None:
        grids.append(("sst", sst_grid, 100.0, -500))  # scale=100, offset=-500 → 0.01°C precision
    if eke_grid is not None:
        grids.append(("eke", eke_grid, 100000.0, 0))  # scale=1e5 → 0.00001 m²/s²

    buf.write(struct.pack("<I", len(grids)))
    for name, grid, scale, offset in grids:
        _pack_grid(buf, name, grid, scale, offset, compression_level)

    payload = buf.getvalue()
    size_kb = len(payload) / 1024

    if len(payload) > MAX_PAYLOAD_BYTES:
        log.warning(f"  ⚠️ Payload {size_kb:.1f}KB exceeds {MAX_PAYLOAD_BYTES // 1024}KB limit!")
    else:
        log.info(f"  📦 Packed: {size_kb:.1f}KB ({len(hotspots)} hotspots"
                 f" + {len(grids)} grids)")

    return payload


def unpack_daily_forecast(data: bytes) -> Dict:
    """
    解包每日預報封包。
    Returns: {"metadata": {}, "hotspots": [], "grids": {"sst": ndarray, ...}}
    """
    buf = io.BytesIO(data)

    magic = buf.read(4)
    if magic != b"OMPK":
        raise ValueError(f"Invalid magic: {magic}")

    version = struct.unpack("<I", buf.read(4))[0]
    flags = struct.unpack("<I", buf.read(4))[0]

    # Meta
    meta_len = struct.unpack("<I", buf.read(4))[0]
    meta_bytes = zlib.decompress(buf.read(meta_len))
    metadata = json.loads(meta_bytes.decode("utf-8"))

    # Hotspots
    hs_len = struct.unpack("<I", buf.read(4))[0]
    hs_bytes = zlib.decompress(buf.read(hs_len))
    hotspots = json.loads(hs_bytes.decode("utf-8"))

    # Grids
    grid_count = struct.unpack("<I", buf.read(4))[0]
    grids = {}
    for _ in range(grid_count):
        name, grid = _unpack_grid(buf)
        grids[name] = grid

    return {
        "metadata": metadata,
        "hotspots": hotspots,
        "grids": grids,
    }


def _slim_hotspots(hotspots: List[Dict], max_spots: int = 30) -> List[Dict]:
    """
    剝離肥大欄位, 只保留航海必需品。
    """
    slim = []
    for h in hotspots[:max_spots]:
        slim.append({
            "r": h.get("rank", 0),
            "la": round(h.get("lat", 0), 3),
            "lo": round(h.get("lon", 0), 3),
            "sp": h.get("species", ""),
            "sc": round(h.get("score", 0), 2),
            "cpue": round(h.get("ml_cpue_kg_day", 0), 1),
            "conf": round(h.get("ml_confidence", 0), 2),
            "eez": h.get("eez", ""),
            "dist": h.get("distance_nm", 0),
            "sst": h.get("sst", None),
            "df": h.get("dominant_features", [])[:3],
        })
    return slim


def _pack_grid(buf, name: str, grid: np.ndarray, scale: float, offset: int, level: int):
    """Quantize float32 → int16, zlib compress, write to buffer."""
    ny, nx = grid.shape
    name_bytes = name.encode("utf-8")

    # Header: name_len, name, ny, nx, scale, offset
    buf.write(struct.pack("<B", len(name_bytes)))
    buf.write(name_bytes)
    buf.write(struct.pack("<II", ny, nx))
    buf.write(struct.pack("<dq", scale, offset))

    # Quantize
    safe = np.where(np.isfinite(grid), grid, 0.0)
    quantized = np.clip((safe * scale + offset), -32768, 32767).astype(np.int16)
    raw_bytes = quantized.tobytes()
    compressed = zlib.compress(raw_bytes, level)
    buf.write(struct.pack("<I", len(compressed)))
    buf.write(compressed)


def _unpack_grid(buf) -> tuple:
    """Read and dequantize a grid from buffer."""
    name_len = struct.unpack("<B", buf.read(1))[0]
    name = buf.read(name_len).decode("utf-8")
    ny, nx = struct.unpack("<II", buf.read(8))
    scale, offset = struct.unpack("<dq", buf.read(16))
    comp_len = struct.unpack("<I", buf.read(4))[0]
    compressed = buf.read(comp_len)
    raw_bytes = zlib.decompress(compressed)
    quantized = np.frombuffer(raw_bytes, dtype=np.int16).reshape(ny, nx)
    grid = (quantized.astype(np.float32) - offset) / scale
    return name, grid


def measure_payload_size(payload: bytes) -> Dict:
    """回傳 payload 大小指標"""
    size_bytes = len(payload)
    return {
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024, 1),
        "under_limit": size_bytes <= MAX_PAYLOAD_BYTES,
        "limit_kb": MAX_PAYLOAD_BYTES // 1024,
    }
