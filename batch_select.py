# -*- coding: utf-8 -*-
import json
import argparse
from pathlib import Path
from datetime import datetime

from Ref_view import build_image_records_from_dirs
from Source_view import select_reference_near_nadir, select_topK_for_ref


# ===================== 全局固定参数（一般不改） =====================

TOPK = 4
DIR_THRESH_DEG = 20.0
W_DISP, H_DISP = 768, 768

# JAX / OMA 区域配置：只在这里维护一次即可
PLACE_CONFIG = {
    "JAX": {
        "img_root": r"H:\MVS-Dataset\US3D-MVS-768\JAX",              # 形如 JAX\068\image
        "imd_root": r"H:\MVS-Dataset\US3D-MVS-768\Track3-Metadata\JAX",
        "rpc_root": None,                                            # 有 RPC 根目录就在这里改
        # 高程范围（可按你前面标注的建议来调）
        # JAX 高度范围 -31.560 到 140.012 建议训练范围设置为：-32 到 224
        "zmin": -32.0,
        "zmax": 32.0,
    },
    "OMA": {
        "img_root": r"H:\MVS-Dataset\US3D-MVS-768\OMA",
        "imd_root": r"H:\MVS-Dataset\US3D-MVS-768\Track3-Metadata\OMA",
        "rpc_root": None,
        # OMA 高度范围 257.149 到 368.628 建议训练范围设置为：128 到 384
        "zmin": 258.0,
        "zmax": 324.0,
    },
}


# 这些全局变量会在 main() 里根据 place 动态赋值
IMG_ROOT: Path
IMD_DIR_ROOT: Path
RPC_DIR_ROOT: Path | None
OUT_ROOT: Path
ZMIN: float
ZMAX: float


def rec_to_small(rec):
    """只保留后续会用到的最小字段（纯基础类型）"""
    return {
        "image_id": getattr(rec, "image_id", None),
        "tif": str(getattr(rec, "raster_path", "") or ""),
        "imd": str(getattr(rec, "imd_path", "") or ""),
        "rpc": str(getattr(rec, "rpc_path", "") or ""),
    }


def pick_to_small(p):
    """兼容 (rec, scoreinfo) / (rec, score) / rec 三种返回"""
    if isinstance(p, (list, tuple)):
        rec = p[0]
        info = p[1] if len(p) > 1 else None   # 现在 info 是那个 row dict
    else:
        rec, info = p, None

    score = None
    if isinstance(info, dict):
        # ★ 优先用带时间权重的最终分数
        cand = info.get("S_pair_time",
                        info.get("S_pair_stable",
                                 info.get("S_pair", None)))
        try:
            score = float(cand) if cand is not None else None
        except Exception:
            score = None
    elif info is not None:
        try:
            score = float(info)
        except Exception:
            score = None

    s = rec_to_small(rec)
    s["score"] = score
    return s



def process_scene(image_dir: Path, place: str):
    """
    处理一个 scene 目录（形如 ...\JAX\068\image）：
      - 读入该 scene 的所有影像记录
      - 选择参考视图
      - 执行 Top-K 源视图筛选
      - 输出一个 JSON 到 OUT_ROOT 下
    """
    scene = image_dir.parent.name  # e.g. '068'

    # 1) 读取记录（仅为选择使用）
    records, _ = build_image_records_from_dirs(
        tif_dir=str(image_dir),
        imd_dir=str(IMD_DIR_ROOT),
        rpc_dir=str(RPC_DIR_ROOT) if RPC_DIR_ROOT else None,
        require_rpc=True,
        parse_imd=True,
    )
    assert records, f"{place} {scene}: records 为空，请检查 IMD/RPC"

    # 2) 选参考
    ref, _ = select_reference_near_nadir(records, Z0=0.0, time_scope="same_group")

    # 3) Top-K
    sources = [r for r in records if r is not ref]
    picked, _ = select_topK_for_ref(
        ref, sources, ZMIN, ZMAX, K=TOPK, dir_thresh_deg=DIR_THRESH_DEG,
        W_disp=W_DISP, H_disp=H_DISP,
        use_content_diversity=True, tiles_xy=(6, 6),
        tile_cov_thresh=0.25, eta_px_per_m_thresh=0.02
    )

    # 4) 只输出需要的最小 JSON
    data = {
        "place": place,
        "scene": scene,
        "time": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "Zmin": ZMIN,
            "Zmax": ZMAX,
            "K": TOPK,
            "dir_thresh_deg": DIR_THRESH_DEG,
            "W_disp": W_DISP,
            "H_disp": H_DISP,
        },
        "ref": rec_to_small(ref),
        "picked": [pick_to_small(p) for p in picked],        # 只保留 image_id/路径/分数
        "all_ids": [r.image_id for r in records],            # 可选：该 scene 下所有可用影像 id
    }

    out_path = OUT_ROOT / f"{place}_{scene}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] {place} {scene} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CasMVS-style JSON for US3D satellite scenes (JAX / OMA)."
    )
    parser.add_argument(
        "--place",
        type=str,
        choices=["JAX", "OMA"],
        default="OMA",
        help="测试区域（JAX 或 OMA），只需要改这个参数即可。",
    )
    args = parser.parse_args()
    place = args.place

    # === 根据 place 选择对应配置 ===
    cfg = PLACE_CONFIG[place]

    global IMG_ROOT, IMD_DIR_ROOT, RPC_DIR_ROOT, OUT_ROOT, ZMIN, ZMAX

    IMG_ROOT = Path(cfg["img_root"])
    IMD_DIR_ROOT = Path(cfg["imd_root"])
    RPC_DIR_ROOT = Path(cfg["rpc_root"]) if cfg.get("rpc_root") else None
    ZMIN, ZMAX = float(cfg["zmin"]), float(cfg["zmax"])

    OUT_ROOT = Path(f"./casmvs_json_{place}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] place={place}")
    print(f"[INFO] IMG_ROOT = {IMG_ROOT}")
    print(f"[INFO] IMD_DIR_ROOT = {IMD_DIR_ROOT}")
    print(f"[INFO] RPC_DIR_ROOT = {RPC_DIR_ROOT}")
    print(f"[INFO] Z range = [{ZMIN}, {ZMAX}]")
    print(f"[INFO] OUT_ROOT = {OUT_ROOT}")

    # === 遍历该区域的所有 scene 目录，形如 ...\JAX\068\image ===
    for image_dir in sorted(IMG_ROOT.glob("*/image")):
        try:
            process_scene(image_dir, place)
        except Exception as e:
            print(f"[FAIL] {place} {image_dir}: {e}")


if __name__ == "__main__":
    main()
