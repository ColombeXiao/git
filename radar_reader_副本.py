"""Decode Q1.1 radar records and freeze a distance-based train/holdout split."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import re
import struct

import numpy as np


N_SAMPLES = 16_000
N_SAMPLES_PER_SWEEP = 4_000
SAMPLE_RATE_HZ = 4_000_000_000
BAND_HZ = (300_000_000, 1_700_000_000)
HEIGHT_CM = (75.5, 68.5, 61.5, 54.5, 47.5, 40.5, 33.5, 26.5)
FILE_PATTERN = re.compile(r"(?P<time>\d{4}(?:-\d{2}){5})_(?P<samples>\d+)_(?P<row>[0-7])\.dat")
GROUP_PATTERN = re.compile(r"data(?P<index>\d+)(?:-(?P<distance>\d+(?:\.\d+)?)cm)?")
MATERIAL_DIRS = {"air": "空场数据", "wall": "墙体数据", "metal": "金属板数据"}
PACKET_PREFIX = struct.Struct("<III")
PACKET_FIELDS = struct.Struct("<fIII")


@dataclass(frozen=True)
class Antenna:
    row: int
    in_row: int
    column: int
    x_cm: float
    z_cm: float

    @property
    def id(self) -> str:
        return f"R{self.row:02d}C{self.column:02d}"


def _antenna(row: int, in_row: int) -> Antenna:
    column = 2 * in_row - (row % 2)
    return Antenna(row, in_row, column, 3.5 * (8 - column), HEIGHT_CM[row - 1])


TX = _antenna(4, 4)
CHANNELS = tuple(
    _antenna(row, in_row)
    for row in range(1, 9)
    for in_row in range(1, 9)
    if (row, in_row) != (TX.row, TX.in_row)
)
RECEIVER_MASK = np.ones(64, dtype=bool)
RECEIVER_MASK[(TX.row - 1) * 8 + TX.in_row - 1] = False


@dataclass(frozen=True)
class Group:
    material: str
    name: str
    distance_cm: float | None
    split: str
    timestamp: str
    files: tuple[Path, ...]

    @property
    def id(self) -> str:
        return f"{self.material}/{self.name}"


def _find_range_dir(project_dir: Path) -> Path:
    relative_path = Path("附件") / "附件2-测距数据"
    direct = project_dir / relative_path
    if direct.is_dir():
        return direct
    candidates = sorted(p for p in project_dir.glob(f"*/{relative_path}") if p.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"预期一个附件2-测距数据目录，找到 {len(candidates)} 个：{project_dir}")
    return candidates[0]


def _discover_capture(material: str, folder: Path) -> tuple[int, Group]:
    info = GROUP_PATTERN.fullmatch(folder.name)
    if info is None:
        raise ValueError(f"无法从目录名解析测量组：{folder}")
    distance = float(info["distance"]) if info["distance"] is not None else None
    if (distance is None) != (material == "air"):
        raise ValueError(f"距离信息与目标类别不符：{folder}")
    rows = {}
    timestamps = set()
    for path in sorted(folder.glob("*.dat")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None or int(match["samples"]) != N_SAMPLES:
            raise ValueError(f"文件名或标称样点数异常：{path}")
        row = int(match["row"])
        if row in rows:
            raise ValueError(f"测量组内有多个相同行编号文件：{path}")
        rows[row] = path
        timestamps.add(match["time"])
    if set(rows) != set(range(8)) or len(timestamps) != 1:
        raise ValueError(f"测量组八行不齐或采集时间不一致：{folder}")
    timestamp = timestamps.pop()
    datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
    files = tuple(rows[row] for row in range(8))
    return int(info["index"]), Group(material, folder.name, distance, "calibration", timestamp, files)


def discover_groups(project_dir: Path) -> list[Group]:
    """Select one singleton distance near the midpoint of each distance quintile.

    The rule uses folder names only. Repeated distances stay together in the
    calibration partition, so no nominal distance appears on both sides.
    """
    root = _find_range_dir(Path(project_dir).resolve())
    result = []
    for material, name in MATERIAL_DIRS.items():
        category_dir = root / name
        captures = [_discover_capture(material, folder) for folder in category_dir.iterdir() if folder.is_dir()]
        groups = [group for _, group in sorted(captures, key=lambda pair: pair[0])]
        expected_count = 10 if material == "air" else 20
        if len(groups) != expected_count:
            raise ValueError(f"{category_dir} 应有 {expected_count} 个测量组，实际 {len(groups)} 个")
        if material != "air":
            counts = Counter(group.distance_cm for group in groups)
            singleton_distances = sorted(distance for distance, count in counts.items() if count == 1)
            if len(singleton_distances) < 5:
                raise ValueError(f"{category_dir} 中可供留出的唯一距离不足 5 个")
            n_distances = len(singleton_distances)
            holdout_distances = {singleton_distances[(2 * i + 1) * n_distances // 10] for i in range(5)}
            groups = [
                replace(group, split="holdout") if group.distance_cm in holdout_distances else group
                for group in groups
            ]
        result.extend(groups)
    return result


def _read_packet_file(path: Path) -> dict[int, np.ndarray]:
    packets = {}
    total_bytes = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < total_bytes:
            offset = stream.tell()
            prefix = stream.read(PACKET_PREFIX.size)
            if len(prefix) != PACKET_PREFIX.size:
                raise ValueError(f"包头截断：{path}，字节 {offset}")
            magic, length, packet_type = PACKET_PREFIX.unpack(prefix)
            remaining = length - 4
            if magic != 963072 or remaining < 0 or remaining > total_bytes - stream.tell():
                raise ValueError(f"包标记或包长异常：{path}，字节 {offset}")
            if packet_type != 102:
                stream.seek(remaining, 1)
                continue
            if remaining < PACKET_FIELDS.size:
                raise ValueError(f"采样包字段长度不足：{path}，字节 {offset}")
            _, packet_id, samples, bytes_per_time = PACKET_FIELDS.unpack(stream.read(PACKET_FIELDS.size))
            data_bytes = samples * bytes_per_time
            if samples != N_SAMPLES or bytes_per_time != 16 or data_bytes != remaining - PACKET_FIELDS.size:
                raise ValueError(f"采样包尺寸异常：{path}，包编号 {packet_id}")
            if packet_id in packets:
                raise ValueError(f"同一行设备包编号重复：{path}，包编号 {packet_id}")
            payload = stream.read(data_bytes)
            if len(payload) != data_bytes:
                raise ValueError(f"采样包数据截断：{path}，包编号 {packet_id}")
            packets[packet_id] = np.frombuffer(payload, dtype="<i2").reshape(N_SAMPLES, 8).T
    return packets


def read_group(group: Group, audit_out: dict | None = None) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return complete matched packets as int16 (packets, receivers, samples).

    Sample number 0 through 15999 is shared across rows of each matched
    packet. Position 0 need not be aligned to a sweep boundary across packets.
    """
    source = [_read_packet_file(path) for path in group.files]
    common = set.intersection(*(set(packets) for packets in source))
    if not common:
        raise ValueError(f"八行文件找不到共同的设备包编号：{group.id}")
    packet_ids = tuple(packet_id for packet_id in source[0] if packet_id in common)
    if audit_out is not None:
        audit_out.update(
            row_packet_ids=[tuple(packets) for packets in source],
            unmatched_packet_ids=[tuple(packet_id for packet_id in packets if packet_id not in common) for packets in source],
        )
    data = np.stack(
        [
            np.concatenate([row[packet_id] for row in source], axis=0)[RECEIVER_MASK]
            for packet_id in packet_ids
        ]
    )
    return data, packet_ids


def inventory_rows(groups: list[Group], project_dir: Path) -> list[dict]:
    """One metadata row per raw file, in group order and row order."""
    project_dir = Path(project_dir).resolve()
    return [
        {
            "group_id": group.id,
            "material": group.material,
            "split": group.split,
            "distance_cm": group.distance_cm,
            "timestamp": group.timestamp,
            "array_row": row,
            "relative_path": str(path.relative_to(project_dir)),
            "size_bytes": path.stat().st_size,
        }
        for group in groups
        for row, path in enumerate(group.files, start=1)
    ]
