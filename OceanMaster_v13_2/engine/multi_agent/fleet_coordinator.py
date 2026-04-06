"""
MAS — Multi-Agent Systems 多智慧體系統（船隊協同作戰）

============================================================
🎯 功能說明：
    讓多艘漁船上的 AI Agent 互相通訊，
    分享漁場情報、協調分工，達成船隊級的最佳捕撈效率。

    就像蜂群智慧：
    - A 船發現好漁場 → 通知所有友船
    - B 船走過空漁場 → 告訴大家別來
    - Agent 自動分配「誰去哪個熱點」，避免所有船擠在同一個點

📌 架構狀態：✅ 完整架構  |  ❌ 尚未部署
📌 缺少什麼：多船通訊基礎設施（衛星通訊 / 無線電數據鏈）
📌 買家需要：至少 2 艘以上的船隻同時運行 OceanMaster
📌 通訊方式：衛星電話數據通道 / Iridium SBD 短訊

📌 與 AI Agent 的關係：
    每艘船上跑一個 FishingAIAgent (fishing_agent.py)，
    FleetCoordinator 負責在所有 Agent 之間傳遞情報。
============================================================
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """船間通訊的訊息類型"""
    HOTSPOT_REPORT = "hotspot_report"       # 「我發現魚了！」
    EMPTY_ZONE_REPORT = "empty_zone"        # 「這裡沒魚，別來」
    WEATHER_ALERT = "weather_alert"         # 「颱風來了！」
    POSITION_UPDATE = "position_update"     # 定期位置回報
    CATCH_REPORT = "catch_report"           # 漁獲量回報
    ASSIGNMENT_ORDER = "assignment_order"   # 協調中心的分配指令


@dataclass
class FleetMessage:
    """船隊通訊訊息"""
    sender_vessel_id: str
    message_type: MessageType
    payload: Dict
    timestamp: str = ""
    priority: int = 0  # 0=一般, 1=重要, 2=緊急

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class VesselState:
    """船隻狀態"""
    vessel_id: str
    vessel_name: str
    lat: float = 0.0
    lon: float = 0.0
    current_catch_tons: float = 0.0
    fuel_remaining_pct: float = 100.0
    assigned_hotspot: Optional[str] = None
    last_update: str = ""
    is_online: bool = True


class FleetCoordinator:
    """
    船隊協調中心

    管理所有船隻的狀態，分配漁場責任區，
    轉發船間情報，避免資源浪費。

    ⚠️ 架構狀態：骨架已完成
    ⚠️ 買家使用流程：
        1. 初始化 FleetCoordinator
        2. 註冊所有船隻
        3. 讓每艘船的 Agent 定期回報狀態
        4. 協調中心自動分配最佳策略

    ⚠️ 通訊實作方式（買家擇一）：
        - WebSocket / MQTT（近海有網路時）
        - Iridium SBD 衛星短訊（遠洋無網路時）
        - VHF 無線電數據鏈（短距離船對船）
    """

    def __init__(self, fleet_name: str = "OceanMaster Fleet"):
        self.fleet_name = fleet_name
        self._vessels: Dict[str, VesselState] = {}
        self._message_queue: List[FleetMessage] = []
        self._shared_intelligence: Dict[str, Dict] = {}  # 共享情報庫

        logger.info(f"船隊協調中心初始化: {fleet_name}")

    def register_vessel(self, vessel_id: str, vessel_name: str, port_lat: float, port_lon: float):
        """
        註冊一艘船

        ⚠️ 範例：
            coordinator.register_vessel("V001", "海豐號", 22.6, 120.3)
            coordinator.register_vessel("V002", "新東洋號", 22.6, 120.3)
        """
        self._vessels[vessel_id] = VesselState(
            vessel_id=vessel_id,
            vessel_name=vessel_name,
            lat=port_lat,
            lon=port_lon,
            last_update=datetime.now().isoformat()
        )
        logger.info(f"船隻已註冊: {vessel_name} ({vessel_id})")

    def receive_message(self, message: FleetMessage):
        """
        接收來自船隻的訊息

        ⚠️ 實際部署時，此函數由通訊模組呼叫
        """
        self._message_queue.append(message)

        # 更新共享情報
        if message.message_type == MessageType.HOTSPOT_REPORT:
            self._shared_intelligence[f"hotspot_{message.timestamp}"] = message.payload

        elif message.message_type == MessageType.EMPTY_ZONE_REPORT:
            self._shared_intelligence[f"empty_{message.timestamp}"] = message.payload

        # 更新船隻位置
        if message.message_type == MessageType.POSITION_UPDATE:
            vid = message.sender_vessel_id
            if vid in self._vessels:
                self._vessels[vid].lat = message.payload.get("lat", 0)
                self._vessels[vid].lon = message.payload.get("lon", 0)
                self._vessels[vid].last_update = message.timestamp

        logger.info(f"收到訊息: {message.message_type.value} from {message.sender_vessel_id}")

    def assign_hotspots(self, hotspots: List[Dict]) -> Dict[str, Dict]:
        """
        自動分配漁場責任區

        使用貪婪法：每艘船分配「離它最近且機率最高」的熱點，
        避免所有船都擠在同一個地方。

        ⚠️ TODO: 買家可以用 genetic_optimizer.py 的 GA 做更好的分配

        Args:
            hotspots: AI 模型輸出的熱點列表
                     [{"lat": ..., "lon": ..., "probability": ...}, ...]

        Returns:
            分配結果 {vessel_id: {"assigned_hotspot": ..., "distance_km": ...}}
        """
        raise NotImplementedError(
            "⚠️ 船隊分配需要：\n"
            "  1. 至少 2 艘已註冊的船隻\n"
            "  2. AI 模型輸出的熱點列表\n"
            f"目前已註冊 {len(self._vessels)} 艘船隻。"
        )

    def get_shared_intelligence(self) -> Dict:
        """取得所有船隻共享的情報"""
        return self._shared_intelligence

    def fleet_status(self) -> Dict:
        """取得整個船隊的狀態摘要"""
        return {
            "fleet_name": self.fleet_name,
            "total_vessels": len(self._vessels),
            "online_vessels": sum(1 for v in self._vessels.values() if v.is_online),
            "vessels": {vid: {
                "name": v.vessel_name,
                "position": (v.lat, v.lon),
                "catch_tons": v.current_catch_tons,
                "fuel_pct": v.fuel_remaining_pct,
                "assigned": v.assigned_hotspot
            } for vid, v in self._vessels.items()},
            "shared_reports": len(self._shared_intelligence),
            "pending_messages": len(self._message_queue)
        }
