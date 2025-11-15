# -*- coding: utf-8 -*-
import math
import csv
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from Satellite_Images import ImageRecord
from View_Selection import build_image_records_from_dirs, select_reference_near_nadir

# ===== Logging =====
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ===== Data structure =====
@dataclass
class PairGeoStats:
    theta_deg: float     # 汇聚角 (IMD az/el 或 RPC-ENU)
    M50_px: float        # 扫掠位移中位数 (像素)
    M95_px: float        # 扫掠位移95分位 (像素)
    cover_ratio: float   # 有效覆盖率 C (0..1)


# ===== WGS84 constants & coordinate tools (scalar-safe) =====
_A = 6378137.0
_F = 1/298.257223563
_E2 = _F*(2-_F)

def _geodetic_to_ecef(lon_deg, lat_deg, h_m):
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    h   = np.asarray(h_m, dtype=np.float64)
    lon, lat, h = np.broadcast_arrays(lon, lat, h)

    sl, cl = np.sin(lat), np.cos(lat)
    N = _A / np.sqrt(1 - _E2*sl*sl)
    X = (N + h) * cl * np.cos(lon)
    Y = (N + h) * cl * np.sin(lon)
    Z = (N*(1-_E2) + h) * sl
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # (N,3)

def _ecef_to_enu(XYZ, lon0_deg, lat0_deg, h0_m):
    XYZ = np.atleast_2d(np.asarray(XYZ, dtype=np.float64))
    X0, Y0, Z0 = _geodetic_to_ecef(lon0_deg, lat0_deg, h0_m)[0]

    lon0 = math.radians(float(lon0_deg)); lat0 = math.radians(float(lat0_deg))
    sl, cl = math.sin(lat0), math.cos(lat0)
    so, co = math.sin(lon0), math.cos(lon0)
    R = np.array([[-so,      co,     0],
                  [-cl*co, -cl*so,  sl],
                  [ sl*co,  sl*so,  cl]], dtype=np.float64)
    d = XYZ - np.array([X0, Y0, Z0], dtype=np.float64)
    return d @ R.T


# ===== Unified RPC wrappers (geodetic semantics) =====
def _rpc_photo2geo(rpc, u, v, Z):
    """
    输入 (u,v,Z) -> (lon,lat,h)，单位：度、度、米。返回 (N,3)
    """
    if hasattr(rpc, "RPC_PHOTO2OBJ"):
        lat, lon = rpc.RPC_PHOTO2OBJ(u, v, Z)   # 常见实现返回 (lat, lon)
        h = np.asarray(Z, dtype=np.float64)
        if h.shape == ():
            h = np.full_like(lon, float(h))
        return np.stack([lon, lat, h], axis=-1)
    elif hasattr(rpc, "photo2obj"):
        X, Y, Z_ = rpc.photo2obj(u, v, Z)
        return np.stack([X, Y, Z_], axis=-1)    # 若你的库是米系XYZ，这里不要用该分支
    else:
        raise RuntimeError("RPC model lacks photo2obj/RPC_PHOTO2OBJ")

def _rpc_geo2photo(rpc, lon, lat, h):
    """
    输入 (lon,lat,h) -> (u,v)；返回 shape=(N,2)
    """
    if hasattr(rpc, "RPC_OBJ2PHOTO"):
        # 注意：大多数实现要求 (lat, lon, h) 传入
        u, v = rpc.RPC_OBJ2PHOTO(lat, lon, h)
        return np.stack([u, v], axis=-1)
    elif hasattr(rpc, "obj2photo"):
        u, v = rpc.obj2photo(lat, lon, h)
        return np.stack([u, v], axis=-1)
    else:
        raise RuntimeError("RPC model lacks obj2photo/RPC_OBJ2PHOTO")


# ===== Convergence angle via IMD az/el =====
def convergence_angle(az_deg_1, az_deg_2, el_deg_1, el_deg_2):
    az1, az2 = map(math.radians, [az_deg_1, az_deg_2])
    el1, el2 = map(math.radians, [el_deg_1, el_deg_2])
    cosd = math.sin(el1)*math.sin(el2) + math.cos(el1)*math.cos(el2)*math.cos(az1 - az2)
    cosd = max(-1.0, min(1.0, cosd))
    return math.degrees(math.acos(cosd))


# ===== Convergence angle via RPC & ENU (fallback) =====
def _convergence_angle_deg_enu(rpc_ref, rpc_src, u0, v0, Zmid):
    dz = 30.0
    # ref
    Pm = _rpc_photo2geo(rpc_ref, np.array([u0]), np.array([v0]), Zmid - dz)[0]
    Pp = _rpc_photo2geo(rpc_ref, np.array([u0]), np.array([v0]), Zmid + dz)[0]
    E_m = _geodetic_to_ecef(Pm[0], Pm[1], Pm[2])
    E_p = _geodetic_to_ecef(Pp[0], Pp[1], Pp[2])
    ENU_m = _ecef_to_enu(E_m, Pm[0], Pm[1], Pm[2])
    ENU_p = _ecef_to_enu(E_p, Pm[0], Pm[1], Pm[2])
    r_ref = (ENU_p - ENU_m)[0]; r_ref /= (np.linalg.norm(r_ref) + 1e-9)
    # src
    Pm = _rpc_photo2geo(rpc_src, np.array([u0]), np.array([v0]), Zmid - dz)[0]
    Pp = _rpc_photo2geo(rpc_src, np.array([u0]), np.array([v0]), Zmid + dz)[0]
    E_m = _geodetic_to_ecef(Pm[0], Pm[1], Pm[2])
    E_p = _geodetic_to_ecef(Pp[0], Pp[1], Pp[2])
    ENU_m = _ecef_to_enu(E_m, Pm[0], Pm[1], Pm[2])
    ENU_p = _ecef_to_enu(E_p, Pm[0], Pm[1], Pm[2])
    r_src = (ENU_p - ENU_m)[0]; r_src /= (np.linalg.norm(r_src) + 1e-9)
    c = float(np.clip(np.dot(r_ref, r_src), -1.0, 1.0))
    return math.degrees(math.acos(c))


# ===== Depth sensitivity η (computed at network input scale) =====
def _depth_sensitivity_eta(rec_ref: ImageRecord, rec_src: ImageRecord,
                           us_disp: np.ndarray, vs_disp: np.ndarray,
                           Zmid: float,
                           W_ref_disp: int, H_ref_disp: int,
                           W_src_disp: int, H_src_disp: int) -> np.ndarray:
    dz = 1.0
    # 显示 -> 原始
    W_ref_full = float(rec_ref.num_cols or W_ref_disp)
    H_ref_full = float(rec_ref.num_rows or H_ref_disp)
    u_ref_full = us_disp * (W_ref_full / W_ref_disp)
    v_ref_full = vs_disp * (H_ref_full / H_ref_disp)

    # 反投 ref
    Pm = _rpc_photo2geo(rec_ref.rpc, u_ref_full, v_ref_full, Zmid - dz)
    Pp = _rpc_photo2geo(rec_ref.rpc, u_ref_full, v_ref_full, Zmid + dz)

    # 前投到 src 原始
    uv_m_full = _rpc_geo2photo(rec_src.rpc, Pm[:,0], Pm[:,1], Pm[:,2])
    uv_p_full = _rpc_geo2photo(rec_src.rpc, Pp[:,0], Pp[:,1], Pp[:,2])

    # 原始 -> 显示（网络尺度）
    W_src_full = float(rec_src.num_cols or W_src_disp)
    H_src_full = float(rec_src.num_rows or H_src_disp)
    ku = W_src_disp / W_src_full
    kv = H_src_disp / H_src_full
    uv_m = np.column_stack([uv_m_full[:,0]*ku, uv_m_full[:,1]*kv])
    uv_p = np.column_stack([uv_p_full[:,0]*ku, uv_p_full[:,1]*kv])

    d = np.linalg.norm(uv_p - uv_m, axis=1) / (2.0*dz + 1e-9)  # px/m
    return d


# ===== Coverage ratio C (at network input scale) =====
def _coverage_ratio_rpc(rec_ref: ImageRecord, rec_src: ImageRecord,
                        W_disp: int, H_disp: int,
                        Zmin: float, Zmax: float,
                        step: int = 32) -> float:
    ys, xs = np.mgrid[step/2:H_disp:step, step/2:W_disp:step]
    us_disp = xs.reshape(-1).astype(np.float64)
    vs_disp = ys.reshape(-1).astype(np.float64)

    # 显示 -> 原始
    W_ref_full = float(rec_ref.num_cols or W_disp)
    H_ref_full = float(rec_ref.num_rows or H_disp)
    u_ref_full = us_disp * (W_ref_full / W_disp)
    v_ref_full = vs_disp * (H_ref_full / H_disp)

    # 反投 & 前投（原始）
    Pmin = _rpc_photo2geo(rec_ref.rpc, u_ref_full, v_ref_full, Zmin)
    Pmax = _rpc_photo2geo(rec_ref.rpc, u_ref_full, v_ref_full, Zmax)
    uv1_full = _rpc_geo2photo(rec_src.rpc, Pmin[:,0], Pmin[:,1], Pmin[:,2])
    uv2_full = _rpc_geo2photo(rec_src.rpc, Pmax[:,0], Pmax[:,1], Pmax[:,2])

    # 原始 -> 显示
    W_src_full = float(rec_src.num_cols or W_disp)
    H_src_full = float(rec_src.num_rows or H_disp)
    ku = W_disp / W_src_full
    kv = H_disp / H_src_full
    uv1 = np.column_stack([uv1_full[:,0]*ku, uv1_full[:,1]*kv])
    uv2 = np.column_stack([uv2_full[:,0]*ku, uv2_full[:,1]*kv])

    def in_img(uv):
        return (uv[:,0] >= 0) & (uv[:,0] < W_disp) & (uv[:,1] >= 0) & (uv[:,1] < H_disp)

    valid = in_img(uv1) & in_img(uv2)
    return float(valid.mean())


# ===== Geometry scoring =====
def compute_pair_geo_stats(rec_ref: ImageRecord,
                           rec_src: ImageRecord,
                           Zmin: float, Zmax: float,
                           sample_step: int = 32,
                           W_disp: int = 768, H_disp: int = 768) -> PairGeoStats:
    Zmid = 0.5*(Zmin + Zmax)

    # θ：优先用 IMD az/el；缺失时回退 RPC-ENU
    if (getattr(rec_ref, "sat_az", None) is not None and getattr(rec_ref, "sat_el", None) is not None and
        getattr(rec_src, "sat_az", None) is not None and getattr(rec_src, "sat_el", None) is not None):
        theta = convergence_angle(rec_ref.sat_az, rec_src.sat_az,
                                  rec_ref.sat_el, rec_src.sat_el)
    else:
        theta = _convergence_angle_deg_enu(rec_ref.rpc, rec_src.rpc, W_disp*0.5, H_disp*0.5, Zmid)

    # η & M 分布
    ys, xs = np.mgrid[sample_step/2:H_disp:sample_step, sample_step/2:W_disp:sample_step]
    us = xs.reshape(-1).astype(np.float64)
    vs = ys.reshape(-1).astype(np.float64)

    eta = _depth_sensitivity_eta(rec_ref, rec_src, us, vs, Zmid, W_disp, H_disp, W_disp, H_disp)  # px/m
    M = eta * (Zmax - Zmin)  # px
    M50 = float(np.percentile(M, 50))
    M95 = float(np.percentile(M, 95))

    # 覆盖率
    C = _coverage_ratio_rpc(rec_ref, rec_src, W_disp, H_disp, Zmin, Zmax, step=sample_step)

    return PairGeoStats(theta_deg=theta, M50_px=M50, M95_px=M95, cover_ratio=C)

def score_geo(stats: PairGeoStats) -> float:
    def norm_theta(t):
        if t < 5 or t > 35: return 0.0
        if 10 <= t <= 30:   return 1.0
        if t < 10:          return (t-5)/(10-5)
        return (35-t)/(35-30)
    def norm_M(m):
        if m < 2 or m > 80: return 0.0
        if 10 <= m <= 40:   return 1.0
        if m < 10:          return (m-2)/(10-2)
        return (80-m)/(80-40)
    g = 0.4*norm_theta(stats.theta_deg) + 0.4*norm_M(stats.M50_px) + 0.2*float(np.clip(stats.cover_ratio,0,1))
    ratio = (stats.M95_px + 1e-6)/(stats.M50_px + 1e-6)
    if ratio > 3.0:
        g *= max(0.0, 1.0 - 0.3*(ratio-3.0))
    return float(np.clip(g, 0.0, 1.0))


# ===== Radiometric consistency (with linear brightness alignment) =====
def _bilinear_sample(img, uv):  # uv: [N,2] in (x=u, y=v)
    h, w = img.shape[:2]
    x = np.clip(uv[:,0], 0, w-1)
    y = np.clip(uv[:,1], 0, h-1)
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0+1, 0, w-1);        y1 = np.clip(y0+1, 0, h-1)
    wa = (x1-x)*(y1-y); wb = (x-x0)*(y1-y); wc = (x1-x)*(y-y0); wd = (x-x0)*(y-y0)
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    return (wa*Ia + wb*Ib + wc*Ic + wd*Id)

def _coarse_overlap_gray(rec_ref: ImageRecord, rec_src: ImageRecord,
                         Zmid: float, down: int = 4, step: int = 8):
    ref = cv2.imread(rec_ref.raster_path, cv2.IMREAD_GRAYSCALE)
    src = cv2.imread(rec_src.raster_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or src is None:
        return None, None, None
    if down > 1:
        ref = cv2.resize(ref, (ref.shape[1]//down, ref.shape[0]//down), interpolation=cv2.INTER_AREA)
        src = cv2.resize(src, (src.shape[1]//down, src.shape[0]//down), interpolation=cv2.INTER_AREA)
    H_ref, W_ref = ref.shape[:2]
    H_src, W_src = src.shape[:2]

    ys, xs = np.mgrid[step/2:H_ref:step, step/2:W_ref:step]
    xs_f = xs.reshape(-1).astype(np.float64)
    ys_f = ys.reshape(-1).astype(np.float64)

    # 显示 -> 原始
    W_ref_full = float(rec_ref.num_cols or W_ref*down)
    H_ref_full = float(rec_ref.num_rows or H_ref*down)
    u_ref_full = xs_f * (W_ref_full / W_ref)
    v_ref_full = ys_f * (H_ref_full / H_ref)

    # 地理 -> src原始 -> src显示
    P_geo = _rpc_photo2geo(rec_ref.rpc, u_ref_full, v_ref_full, Zmid)
    uv_src_full = _rpc_geo2photo(rec_src.rpc, P_geo[:,0], P_geo[:,1], P_geo[:,2])

    W_src_full = float(rec_src.num_cols or W_src*down)
    H_src_full = float(rec_src.num_rows or H_src*down)
    ku = W_src / W_src_full
    kv = H_src / H_src_full
    uv_src_disp = np.column_stack([uv_src_full[:,0]*ku, uv_src_full[:,1]*kv])

    mask = (uv_src_disp[:,0]>=0) & (uv_src_disp[:,0]<W_src) & \
           (uv_src_disp[:,1]>=0) & (uv_src_disp[:,1]<H_src)
    n_valid = int(mask.sum())
    if n_valid < 64:
        return None, None, None

    ref_samp = ref[ys_f[mask].astype(np.int32), xs_f[mask].astype(np.int32)].astype(np.float32)
    src_samp = _bilinear_sample(src.astype(np.float32), uv_src_disp[mask])
    return ref_samp, src_samp, n_valid

def _jsd_from_hist(a, b, bins=64):
    ha, _ = np.histogram(a, bins=bins, range=(0,255), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(0,255), density=True)
    ha = ha/(ha.sum()+1e-9); hb = hb/(hb.sum()+1e-9)
    m = 0.5*(ha+hb)
    def _kl(p, q):
        msk = (p>0) & (q>0)
        return float((p[msk]*np.log((p[msk]+1e-12)/(q[msk]+1e-12))).sum())
    return 0.5*_kl(ha, m) + 0.5*_kl(hb, m)

def _fit_linear_affine(x, y):
    # 拟合 y' = a*x + b（最小二乘）
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    X = np.concatenate([x, np.ones_like(x)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a = float(coef[0,0]); b = float(coef[1,0])
    return a, b

def compute_pair_rad_stats(rec_ref: ImageRecord, rec_src: ImageRecord, Zmid: float) -> Dict[str, float]:
    ref_s, src_s, n = _coarse_overlap_gray(rec_ref, rec_src, Zmid, down=4, step=8)
    if ref_s is None:
        return dict(SSIM=0.0, MI_n=0.0, JSD=1.0, dH=1.0)

    # 先标准化到近似 8bit 以稳住直方图统计
    def _norm(z):
        z = (z - z.mean()) / (z.std()+1e-6)
        z = 127.5 + 40*z
        return np.clip(z, 0, 255)

    a = _norm(ref_s).astype(np.float64)
    b = _norm(src_s).astype(np.float64)

    # 线性亮度对齐：b' = g*b + c
    g, c = _fit_linear_affine(b, a)
    b_aligned = np.clip(g*b + c, 0, 255)

    # 相关→SSIM近似
    corr = np.corrcoef(a, b_aligned)[0,1]
    SSIM = float(np.clip((corr+1)/2, 0, 1))

    # 互信息（归一化）
    H2, _, _ = np.histogram2d(a, b_aligned, bins=64, range=[[0,255],[0,255]], density=True)
    px = H2.sum(axis=1); py = H2.sum(axis=0)
    Hx = -(px[px>0]*np.log(px[px>0])).sum()
    Hy = -(py[py>0]*np.log(py[py>0])).sum()
    Hxy = -(H2[H2>0]*np.log(H2[H2>0])).sum()
    MI = float(Hx + Hy - Hxy)
    MI_n = float(np.clip(MI / (min(Hx,Hy)+1e-9), 0, 1))

    # 直方图差异
    JSD = float(_jsd_from_hist(a, b_aligned, bins=64))

    # 熵差
    def _entropy(x, bins=64):
        h, _ = np.histogram(x, bins=bins, range=(0,255), density=True)
        h = h/(h.sum()+1e-9)
        m = h[h>0]
        return float(-(m*np.log(m)).sum())

    dH  = abs(_entropy(a, 64) - _entropy(b_aligned, 64))
    return dict(SSIM=SSIM, MI_n=MI_n, JSD=JSD, dH=dH)

def score_rad(stats: Dict[str, float]) -> float:
    SSIM = stats["SSIM"]; MI = stats["MI_n"]; JSD = stats["JSD"]; dH = stats["dH"]
    S = 0.4*SSIM + 0.3*MI + 0.2*math.exp(-5*JSD) + 0.1*math.exp(-abs(dH))
    return float(np.clip(S, 0.0, 1.0))


# ===== Incoming azimuth (ENU) =====
def _incoming_azimuth_deg(rpc_ref, rpc_src, u0, v0, Zmid):
    dz = 30.0
    Pm = _rpc_photo2geo(rpc_src, np.array([u0]), np.array([v0]), Zmid - dz)[0]
    Pp = _rpc_photo2geo(rpc_src, np.array([u0]), np.array([v0]), Zmid + dz)[0]
    E_m = _geodetic_to_ecef(Pm[0], Pm[1], Pm[2])
    E_p = _geodetic_to_ecef(Pp[0], Pp[1], Pp[2])
    ENU_m = _ecef_to_enu(E_m, Pm[0], Pm[1], Pm[2])
    ENU_p = _ecef_to_enu(E_p, Pm[0], Pm[1], Pm[2])
    v = (ENU_p - ENU_m)[0]  # [E,N,U]
    az = math.degrees(math.atan2(v[0], v[1]))  # 北为0°、顺时针
    return az + 360.0 if az < 0 else az


# ===== 稳定性评估：OFF/Z/尺度扰动 =====
class OffsetRPC:
    """对 RPC 调用注入像素级偏移（模拟 OFF/crop 误差），不改动原始 RPC。"""
    def __init__(self, base_rpc, du=0.0, dv=0.0):
        self.base = base_rpc
        self.du = float(du); self.dv = float(dv)
    # 反投：给输入像素加偏移
    def RPC_PHOTO2OBJ(self, insamp, inline, inhei):
        return self.base.RPC_PHOTO2OBJ(np.asarray(insamp)+self.du,
                                       np.asarray(inline)+self.dv,
                                       inhei)
    # 正投：在输出像素上加偏移
    def RPC_OBJ2PHOTO(self, inlat, inlon, inhei):
        u, v = self.base.RPC_OBJ2PHOTO(inlat, inlon, inhei)
        return np.asarray(u)+self.du, np.asarray(v)+self.dv

def _score_pair_once(rec_ref, rec_src, Zmin, Zmax, W_disp, H_disp):
    gstats = compute_pair_geo_stats(rec_ref, rec_src, Zmin, Zmax, W_disp=W_disp, H_disp=H_disp)
    S_geo  = score_geo(gstats)
    Zmid   = 0.5*(Zmin+Zmax)
    rstats = compute_pair_rad_stats(rec_ref, rec_src, Zmid)
    S_rad  = score_rad(rstats)
    S_pair = 0.5*S_geo + 0.5*S_rad
    return S_pair, S_geo, S_rad

def compute_stability_penalty(rec_ref: ImageRecord, rec_src: ImageRecord,
                              Zmin: float, Zmax: float,
                              W_disp: int=768, H_disp: int=768,
                              off_px: float=0.5,      # OFF 扰动幅度（像素）
                              dZ: float=2.0,          # 深度窗扰动（米）
                              scale_jitter: float=0.1,# 尺度扰动比例（±10%）
                              trials: int=4) -> float:
    """
    返回 [0,1] 的惩罚系数 P，越大说明越不稳定。建议最终打分乘以 (1 - λ*P)。
    """
    # 基准
    S0, _, _ = _score_pair_once(rec_ref, rec_src, Zmin, Zmax, W_disp, H_disp)
    if S0 <= 1e-6:
        return 1.0

    scores = []
    rng = np.random.default_rng(0xC0FFEE)
    for _ in range(trials):
        du = rng.uniform(-off_px, off_px)
        dv = rng.uniform(-off_px, off_px)
        z_bias = rng.uniform(-dZ, dZ)
        s_jit = 1.0 + rng.uniform(-scale_jitter, scale_jitter)

        # 包装 RPC（仅用于本次评估）
        ref_rpc_backup = rec_ref.rpc
        src_rpc_backup = rec_src.rpc
        rec_ref.rpc = OffsetRPC(ref_rpc_backup, du=du, dv=dv)
        rec_src.rpc = OffsetRPC(src_rpc_backup, du=-du, dv=-dv)  # 反向偏移，模拟双侧误差

        # 尺度抖动：只在显示尺度上改（几何采样网格变动）
        Wj = max(64, int(W_disp * s_jit))
        Hj = max(64, int(H_disp * s_jit))

        Sj, _, _ = _score_pair_once(rec_ref, rec_src, Zmin+z_bias, Zmax+z_bias, Wj, Hj)
        scores.append(Sj)

        # 还原
        rec_ref.rpc = ref_rpc_backup
        rec_src.rpc = src_rpc_backup

    scores = np.asarray(scores, dtype=np.float64)
    # 不稳定度 = 相对方差与相对偏差的折中
    rel_var = float(np.var(scores) / (S0*S0 + 1e-9))
    rel_dev = float(np.mean(np.abs(scores - S0)) / (S0 + 1e-9))
    P = float(np.clip(0.5*rel_var + 0.5*rel_dev, 0.0, 1.0))
    return P


# ===== Top-K selection =====
STAB_LAMBDA = 0.35  # 稳定性惩罚权重

def select_topK_for_ref(rec_ref: ImageRecord,
                        sources: List[ImageRecord],
                        Zmin: float, Zmax: float,
                        K: int = 6,
                        dir_thresh_deg: float = 20.0,
                        W_disp: int = 768, H_disp: int = 768,
                        use_content_diversity: bool = False,
                        tiles_xy: Tuple[int,int]=(6,6),
                        tile_cov_thresh: float = 0.25,
                        eta_px_per_m_thresh: float = 0.02) -> List[Tuple[ImageRecord, dict]]:
    Zmid = 0.5*(Zmin + Zmax)
    rows = []

    # 可选：内容多样性所需的 tile 网格（仅用于指标聚合）
    if use_content_diversity:
        Ht, Wt = tiles_xy
        ys, xs = np.mgrid[0:Ht, 0:Wt]
        tile_boxes = []
        for ty in range(Ht):
            for tx in range(Wt):
                u0 = tx * (W_disp / Wt)
                v0 = ty * (H_disp / Ht)
                u1 = (tx+1) * (W_disp / Wt)
                v1 = (ty+1) * (H_disp / Ht)
                tile_boxes.append((int(u0), int(v0), int(u1), int(v1)))

    def _tile_coverage(rec_ref, rec_src):
        # 稀疏采样统计每个 tile 内 η≥阈值 的占比（乘全局可见率作为权重）
        Ht, Wt = tiles_xy
        cov = np.zeros(Ht*Wt, dtype=np.float32)
        step = max(8, int(min(W_disp, H_disp) / 48))
        ys, xs = np.mgrid[step/2:H_disp:step, step/2:W_disp:step]
        us = xs.reshape(-1).astype(np.float64)
        vs = ys.reshape(-1).astype(np.float64)
        eta = _depth_sensitivity_eta(rec_ref, rec_src, us, vs, Zmid, W_disp, H_disp, W_disp, H_disp)
        tx = np.clip((us / (W_disp / Wt)).astype(int), 0, Wt-1)
        ty = np.clip((vs / (H_disp / Ht)).astype(int), 0, Ht-1)
        tid = ty*Wt + tx

        vis_global = _coverage_ratio_rpc(rec_ref, rec_src, W_disp, H_disp, Zmin, Zmax, step=step)
        ok = (eta >= eta_px_per_m_thresh)
        for i in np.where(ok)[0]:
            cov[tid[i]] += 1.0
        tile_counts = np.bincount(tid, minlength=Ht*Wt)
        tile_counts[tile_counts==0] = 1
        cov = cov / tile_counts
        cov *= float(vis_global)
        return cov  # [Ht*Wt]

    for src in sources:
        try:
            gstats = compute_pair_geo_stats(rec_ref, src, Zmin, Zmax, W_disp=W_disp, H_disp=H_disp)
            S_geo  = score_geo(gstats)
            rstats = compute_pair_rad_stats(rec_ref, src, Zmid)
            S_rad  = score_rad(rstats)
            S_pair = 0.5*S_geo + 0.5*S_rad

            # 稳定性惩罚
            P_stab = compute_stability_penalty(rec_ref, src, Zmin, Zmax, W_disp=W_disp, H_disp=H_disp)
            S_pair_stable = float(np.clip(S_pair * (1.0 - STAB_LAMBDA * P_stab), 0.0, 1.0))

            az = _incoming_azimuth_deg(rec_ref.rpc, src.rpc, W_disp*0.5, H_disp*0.5, Zmid)

            row = dict(src=src, S_geo=S_geo, S_rad=S_rad, S_pair=S_pair,
                       S_pair_stable=S_pair_stable, P_stab=P_stab,
                       theta=gstats.theta_deg, M50=gstats.M50_px, M95=gstats.M95_px,
                       C=gstats.cover_ratio, azimuth=az, **rstats)

            if use_content_diversity:
                row["tile_cov"] = _tile_coverage(rec_ref, src)

            rows.append(row)
        except Exception as e:
            logging.warning("pair失败 %s -> %s", getattr(src, "image_id", "<unk>"), e)

    # 先按稳定后的分数排序
    rows.sort(key=lambda r: r["S_pair_stable"], reverse=True)

    picked = []
    if use_content_diversity:
        Ht, Wt = tiles_xy
        covered = np.zeros(Ht*Wt, dtype=bool)

    for r in rows:
        ok = True
        # 维持原“方向多样性 + 扫掠相近抑制”
        for q in picked:
            dphi = abs(r["azimuth"] - q["azimuth"])
            dphi = min(dphi, 360.0 - dphi)
            if dphi < dir_thresh_deg and abs(r["M50"] - q["M50"]) <= 0.2*max(1.0, q["M50"]):
                ok = False; break

        # 内容多样性（可选）
        if use_content_diversity and ok:
            tc = r["tile_cov"]
            inc = (tc >= tile_cov_thresh) & (~covered)
            if ok and inc.any():
                covered[inc] = True

        if ok:
            picked.append(r)
        if len(picked) >= K:
            break

    return [(r["src"], r) for r in picked], rows


# ===== Export =====
def save_pairs_and_scores(out_dir: Path, ref: ImageRecord,
                          picked: List[Tuple[ImageRecord, dict]],
                          all_rows: List[dict],
                          Zmin: float, Zmax: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    # pairs.txt
    with open(out_dir/"pairs.txt", "a", encoding="utf-8") as f:
        line = ref.image_id + " " + " ".join([p[0].image_id for p in picked]) + "\n"
        f.write(line)
    # depth_range
    with open(out_dir/"depth_range.txt", "a", encoding="utf-8") as f:
        f.write(f"{ref.image_id} {Zmin:.3f} {Zmax:.3f}\n")
    # scores.csv
    is_new = not (out_dir/"scores.csv").exists()
    with open(out_dir/"scores.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["ref_id","src_id","theta","M50","M95","C","SSIM","MI_n","JSD","dH",
                        "S_geo","S_rad","S_pair","S_pair_stable","P_stab","azimuth_deg","picked"])
        for r in all_rows:
            w.writerow([ref.image_id, r["src"].image_id, f"{r['theta']:.2f}", f"{r['M50']:.2f}",
                        f"{r['M95']:.2f}", f"{r['C']:.3f}", f"{r['SSIM']:.3f}", f"{r['MI_n']:.3f}",
                        f"{r['JSD']:.4f}", f"{r['dH']:.3f}", f"{r['S_geo']:.3f}", f"{r['S_rad']:.3f}",
                        f"{r['S_pair']:.3f}", f"{r.get('S_pair_stable', r['S_pair']):.3f}",
                        f"{r.get('P_stab', 0.0):.3f}", f"{r['azimuth']:.1f}",
                        1 if any(p[0].image_id==r["src"].image_id for p in picked) else 0])

def sanity_check_rpc_closure(rec: ImageRecord, Zmid: float = 0.0, n_samples: int = 200):
    """
    Quick sanity-check: 随机采样像素 (u,v)，反投到地面再前投回来，计算闭环误差。
    用于验证 RPC/函数接口的正确性。
    """
    if rec.rpc is None:
        print(f"[WARN] {rec.image_id}: 无 RPC")
        return None

    W = 768
    H = 768
    us = np.random.uniform(0, W, size=n_samples)
    vs = np.random.uniform(0, H, size=n_samples)
    Zs = np.full_like(us, Zmid)

    # 反投 + 前投
    P_geo = _rpc_photo2geo(rec.rpc, us, vs, Zs)
    uv_back = _rpc_geo2photo(rec.rpc, P_geo[:,0], P_geo[:,1], P_geo[:,2])

    diff = np.linalg.norm(uv_back - np.stack([us, vs], axis=1), axis=1)
    print(f"[CHECK] {rec.image_id}: mean={diff.mean():.3f}px, median={np.median(diff):.3f}px, max={diff.max():.3f}px")

    if diff.mean() > 1.0:
        print("⚠️ 可能存在单位或坐标系错误（经纬度顺序、米↔度、RPC方向等）")
    return diff


# ===== Main =====
if __name__ == "__main__":
    # 你可以切换到任意子目录
    tif_dir = r"H:\MVS-Dataset\US3D-MVS-768\JAX\068\image"
    imd_dir = r"H:\MVS-Dataset\US3D-MVS-768\Track3-Metadata\JAX"
    rpc_dir = None

    records, logs = build_image_records_from_dirs(
        tif_dir=tif_dir, imd_dir=imd_dir, rpc_dir=rpc_dir,
        require_rpc=True, parse_imd=True
    )
    assert records, "records 为空，请检查 IMD/RPC 解析"

    # 选择参考
    ref, ref_logs = select_reference_near_nadir(records, Z0=0.0, time_scope="same_group")
    # 校验RPC闭环误差
    # sanity_check_rpc_closure(ref, Zmid=-20)

    # height_range = None
    # if args.place == "JAX":
    #     height_range = [-32, 224]
    # elif args.place == "OMA":
    #     height_range = [128, 384]
    # elif args.place == "JAX+OMA":
    #     height_range = [-32, 384]
    # OMA 高度范围 257.149 到 368.628 建议训练范围设置为：128 到 384 JAX 高度范围 -31.560 到 140.012 建议训练范围设置为：-32 到 224

    # 深度窗（例）
    Zmin, Zmax = -32.0, 32.0

    # Top-K（可打开内容多样性）
    sources = [r for r in records if r is not ref]
    picked, all_rows = select_topK_for_ref(
        ref, sources, Zmin, Zmax, K=6, dir_thresh_deg=20.0,
        W_disp=768, H_disp=768,
        use_content_diversity=True,      # ← 若要开启内容多样性，改为 True
        tiles_xy=(6,6),
        tile_cov_thresh=0.25,
        eta_px_per_m_thresh=0.02
    )

    out_dir = Path("./casmvs_inputs")
    save_pairs_and_scores(out_dir, ref, picked, all_rows, Zmin, Zmax)

    logging.info("参考影像: %s", ref.image_id)
    logging.info("选中 source: %s", [p[0].image_id for p in picked])
    logging.info("输出目录: %s", str(out_dir))
