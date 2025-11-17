from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import logging
from RPCM.rpc_core import RPCModelParameter,load_rpc_as_array
from Satellite_Images import ImageRecord
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def _last_number_key(p: Path) -> Optional[int]:
    """
    提取文件名中最后一个连续数字串作为编号键（去前导零），无数字返回 None。
    例：JAX_004_006_RGB.tif -> 6； 06.IMD -> 6
    """
    m = re.findall(r'(\d+)', p.stem)
    if not m:
        return None
    return int(m[-1])  # 去前导零

def _index_imd_files(imd_dir: Path) -> Dict[int, Path]:
    """把 IMD 目录按编号建索引：{编号: IMD路径}，大小写都支持"""
    idx: Dict[int, Path] = {}
    for p in imd_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".imd":
            k = _last_number_key(p)
            if k is not None:
                idx[k] = p
    return idx

def _index_rpc_files(rpc_dir: Path) -> Dict[str, Path]:
    """按 basename(stem) 索引 RPC：{无扩展名: 路径}；大小写都支持"""
    idx: Dict[str, Path] = {}
    for p in rpc_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".rpc":
            idx[p.stem] = p
    return idx

def _find_rpc_for_tif(tif_path: Path, rpc_index: Dict[str, Path]) -> Optional[Path]:
    """
    先同名查找，再允许 stem 的常见变体失败；若 rpc_index 为空，尝试同目录同名 .rpc
    """
    # 1) 直接同名（最稳妥）
    if tif_path.stem in rpc_index:
        return rpc_index[tif_path.stem]
    # 2) 同目录兜底
    cand = tif_path.with_suffix(".rpc")
    if cand.exists():
        return cand
    cand = tif_path.with_suffix(".RPC")
    if cand.exists():
        return cand
    return None

def _load_rpc_param(rpc_path: Path) -> RPCModelParameter:
    # rpc1,_,_ = load_rpc_as_array(str(rpc_path))
    rpc = RPCModelParameter()
    rpc.load_dirpc_from_file(str(rpc_path))   # 若你的文件是 dirpc 格式，改成 load_dirpc_from_file
    return rpc

def build_image_records_from_dirs(
    tif_dir: str | Path,
    imd_dir: Optional[str | Path] = None,
    rpc_dir: Optional[str | Path] = None,
    require_rpc: bool = True,
    parse_imd: bool = True,
) -> Tuple[List[ImageRecord], List[Dict]]:
    tif_dir = Path(tif_dir)
    assert tif_dir.is_dir(), f"TIF 目录不存在：{tif_dir}"

    imd_index: Dict[int, Path] = {}
    if imd_dir:
        imd_index = _index_imd_files(Path(imd_dir))

    rpc_index: Dict[str, Path] = {}
    if rpc_dir:
        rpc_index = _index_rpc_files(Path(rpc_dir))

    tifs = [p for p in tif_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".tif" and not p.name.lower().endswith(".tif.enp")]

    records: List[ImageRecord] = []
    logs: List[Dict] = []

    for tif in sorted(tifs):
        entry = {"tif": str(tif), "rpc": None, "imd": None, "status": "OK", "note": ""}
        try:
            key = _last_number_key(tif)

            # 找 IMD（按编号）
            imd_path = None
            if imd_index and key is not None:
                imd_path = imd_index.get(key)
            if imd_path:
                entry["imd"] = str(imd_path)

            # 找 RPC（索引或同目录兜底）
            rpc_path = _find_rpc_for_tif(tif, rpc_index)
            if rpc_path:
                entry["rpc"] = str(rpc_path)
            elif require_rpc:
                entry["status"] = "SKIP"
                entry["note"] = "缺少匹配 RPC"
                logs.append(entry)
                logging.warning("跳过（无 RPC）：%s", tif.name)
                continue

            # 载入 RPC（若找到）
            rpc_param = _load_rpc_param(rpc_path) if rpc_path else RPCModelParameter()

            rec = ImageRecord(
                image_id=tif.stem,
                raster_path=str(tif),
                rpc=rpc_param,
                imd_path=str(imd_path) if imd_path else None,
            )
            # --- 关键补充：写回 rpc_path，便于后续使用/打印 ---
            if rpc_path:
                rec.rpc_path = str(rpc_path)

            # --- 关键补充：如需要并且提供了 IMD，显式解析 ---
            if parse_imd and imd_path:
                rec.parse_imd()

            records.append(rec)
            logs.append(entry)
            logging.info("OK: %s  (IMD:%s, RPC:%s, key=%s)",
                         tif.name,
                         "Y" if imd_path else "N",
                         "Y" if rpc_path else "N",
                         key)
        except Exception as e:
            entry["status"] = "ERROR"
            entry["note"] = repr(e)
            logs.append(entry)
            logging.error("失败：%s -> %s", tif.name, e)

    return records, logs

def _estimate_offnadir_deg(rec: ImageRecord, Z0: float = 0.0, dh: float = 60.0) -> Optional[float]:
    if rec.off_nadir_meta is not None:
        try:
            return float(rec.off_nadir_meta)
        except Exception:
            pass
    rpc = getattr(rec, "rpc", None)
    if rpc is None:
        return None
    # 尺寸
    if rec.image_size:
        H, W = rec.image_size
    elif hasattr(rpc, "image_size") and rpc.image_size not in (None, (0, 0)):
        H, W = rpc.image_size
    else:
        H, W = 4096, 4096
    # 中心点两层高程
    us = np.array([W * 0.5], dtype=np.float64)
    vs = np.array([H * 0.5], dtype=np.float64)
    Zm = np.full_like(us, Z0 - dh, dtype=np.float64)
    Zp = np.full_like(us, Z0 + dh, dtype=np.float64)
    if hasattr(rpc, "RPC_PHOTO2OBJ"):
        latm, lonm = rpc.RPC_PHOTO2OBJ(us, vs, Zm)
        latp, lonp = rpc.RPC_PHOTO2OBJ(us, vs, Zp)
        lat2m = 111_320.0
        lon2m = 111_320.0 * math.cos(math.radians(float(np.mean([latm, latp]))))
        dx = (lonp - lonm) * lon2m
        dy = (latp - latm) * lat2m
        dz = (Zp - Zm)
    elif hasattr(rpc, "photo2obj"):
        Xm, Ym, Zm_ = rpc.photo2obj(us, vs, Zm)
        Xp, Yp, Zp_ = rpc.photo2obj(us, vs, Zp)
        dx = Xp - Xm; dy = Yp - Ym; dz = Zp_ - Zm_
    else:
        return None
    v = np.stack([dx, dy, dz], axis=-1)
    n = np.linalg.norm(v, axis=1) + 1e-9
    cosang = np.clip(np.abs(v[:, 2]) / n, 0.0, 1.0)
    theta = np.degrees(np.arccos(cosang))
    return float(np.median(theta))

def _score_for_ref(theta_deg: float, cov: float, tex: Optional[float], cloud_ok: Optional[float],
                   time_score: float,
                   theta_cap: float = 25.0,
                   w=(0.55, 0.15, 0.05, 0.05, 0.20)) -> float:
    """
    w 顺序： [nadir, coverage, texture, cloud, time]
    """
    w1, w2, w3, w4, w5 = w
    snadir = 1.0 - min(theta_deg / theta_cap, 1.0)
    scov   = float(np.clip(cov, 0.0, 1.0))
    stex   = float(np.clip(tex, 0.0, 1.0)) if tex is not None else 0.5
    scloud = float(np.clip(cloud_ok, 0.0, 1.0)) if cloud_ok is not None else 0.7
    stime  = float(np.clip(time_score, 0.0, 1.0))
    return w1*snadir + w2*scov + w3*stex + w4*scloud + w5*stime

def _median_time_gap_hours(candidate: ImageRecord,
                           peers: List[ImageRecord]) -> Optional[float]:
    if candidate.acq_time is None:
        return None
    gaps = []
    t0 = candidate.acq_time
    for p in peers:
        if p is candidate:
            continue
        if p.acq_time is None:
            continue
        gaps.append(abs((p.acq_time - t0).total_seconds()) / 3600.0)
    if not gaps:
        return None
    return float(np.median(gaps))

def _peer_scope(records: List[ImageRecord], ref: ImageRecord, scope: str) -> List[ImageRecord]:
    """
    scope:
      - "same_group": 仅同 orbit_id 或 strip_id 的 peers（优先，同组更可比）
      - "all": 全部 records
    """
    if scope == "all":
        return records
    # same_group
    same = []
    for r in records:
        if r is ref:
            continue
        ok = False
        if ref.orbit_id and r.orbit_id and r.orbit_id == ref.orbit_id:
            ok = True
        if ref.strip_id and r.strip_id and r.strip_id == ref.strip_id:
            ok = True
        if ok:
            same.append(r)
    # 若同组为空，退化为全部
    return same if same else records

def select_reference_near_nadir(
    records: List[ImageRecord],
    aoi_poly_xy=None,
    gray_map: Optional[Dict[str, np.ndarray]] = None,
    Z0: float = 0.0,
    hard_theta: float = 35.0,
    min_cov: float = 0.6,
    max_cloud: float = 0.3,
    time_scope: str = "same_group",     # "same_group" 或 "all"
    time_cap_hours: float = 24.0,       # 时间分数封顶尺度（越小越严格）
    time_hard_hours: Optional[float] = None,  # 若设置：median_gap 超过则直接淘汰
    weights=(0.55, 0.15, 0.05, 0.05, 0.20),   # [nadir,cov,tex,cloud,time]
) -> Tuple[ImageRecord, List[Tuple[str, float, dict]]]:
    """
    在原有基础上加入“时间凝聚度”：
      S_time = 1 - min(median_gap_hours / time_cap_hours, 1)
    并可用 time_hard_hours 做硬筛（可不设）。
    """
    logs = []
    cands = []

    # 预先计算 coverage、texture、theta（如你已有方法可直接调用；这里保持鲁棒）
    for rec in records:
        # 覆盖率
        cov = 1.0
        if aoi_poly_xy is not None:
            try:
                cov = rec.coverage_ratio(aoi_poly_xy)
            except Exception:
                cov = 1.0

        # 云
        cloud = rec.cloud_ratio if rec.cloud_ratio is not None else 0.0
        cloud_ok = 1.0 - cloud

        # 离轴角
        theta = _estimate_offnadir_deg(rec, Z0=Z0)
        if theta is None:
            theta = 90.0

        # 纹理（可选）
        tex = None
        if gray_map is not None and rec.image_id in gray_map and hasattr(rec, "texture_score"):
            try:
                tex_raw = float(rec.texture_score(gray_map[rec.image_id]))
                tex = float(np.clip(tex_raw / (tex_raw + 5.0), 0.0, 1.0))
            except Exception:
                tex = None

        # 先做非时间硬筛
        hard_pass = True
        if theta > hard_theta:          hard_pass = False
        if cov   < min_cov:             hard_pass = False
        if cloud > max_cloud:           hard_pass = False

        # 时间 peers 集
        peers = _peer_scope(records, rec, time_scope)
        med_gap = _median_time_gap_hours(rec, peers)
        # 时间硬筛（可选）
        if hard_pass and time_hard_hours is not None and med_gap is not None:
            if med_gap > time_hard_hours:
                hard_pass = False

        # 时间分数
        if med_gap is None:
            # 没有时间信息时给个保守中等分
            s_time = 0.5
        else:
            s_time = 1.0 - min(med_gap / max(time_cap_hours, 1e-6), 1.0)

        score = _score_for_ref(theta_deg=theta, cov=cov, tex=tex, cloud_ok=cloud_ok,
                               time_score=s_time, theta_cap=25.0, w=weights)

        info = dict(theta_deg=theta, cov=cov, cloud=cloud, tex=tex,
                    time_scope=time_scope, median_gap_hours=med_gap, S_time=s_time,
                    hard_pass=hard_pass)
        logs.append((rec.image_id or "<none>", score, info))

        if hard_pass:
            cands.append((rec, score, info))

    # 全部被时间或其他条件筛掉：退化至“离轴角最小”
    if not cands:
        fallback = sorted(
            [(rec, _estimate_offnadir_deg(rec, Z0=Z0) or 90.0) for rec in records],
            key=lambda x: x[1]
        )
        ref = fallback[0][0]
        logs_sorted = sorted(logs, key=lambda x: x[1], reverse=True)
        return ref, logs_sorted

    # 综合排序：分数降序；再按 theta 升序；再按时间中位差升序；最后按采集时间（更早优先）
    def _key(t):
        rec, score, info = t
        theta = info["theta_deg"]
        medg  = info["median_gap_hours"] if info["median_gap_hours"] is not None else 1e6
        tstamp = rec.acq_time.timestamp() if rec.acq_time else 9e18
        return (-score, theta, medg, tstamp)

    cands.sort(key=_key)
    ref = cands[0][0]
    logs_sorted = sorted(logs, key=lambda x: x[1], reverse=True)
    return ref, logs_sorted


if __name__ == "__main__":
    tif_dir = r"H:\MVS-Dataset\US3D-MVS-768\JAX\004\image"
    imd_dir = r"H:\MVS-Dataset\US3D-MVS-768\Track3-Metadata\JAX"  # 你截图里的 IMD/RPB 目录
    rpc_dir = None  # 你的 .rpc 就在 tif 同目录，可为 None

    records, logs = build_image_records_from_dirs(
        tif_dir=tif_dir,
        imd_dir=imd_dir,
        rpc_dir=rpc_dir,
        require_rpc=True,
        parse_imd=True
    )

    print("构建成功条数:", len(records))
    print("首条:", records[0].image_id, records[0].rpc_path)
    # 若 from_wv_imd 写了元数据解析，可以检查：
    print("off_nadir:", records[0].off_nadir_meta, "acq_time:", records[0].acq_time)

