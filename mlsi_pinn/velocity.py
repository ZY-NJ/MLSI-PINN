"""Photospheric velocity from the observed reference absorption line.

The fitting procedure follows FISSPy. The returned velocity uses the model
sign convention, with positive velocity shifting the spectrum blueward.
"""

import numpy as np
import torch
from scipy.signal import savgol_filter

C_KMS = 299792.458


def _get_centerWV_local(line: str) -> float:
    ll = line.lower()
    if ll == "ca":
        return 8542.091
    raise ValueError(f"Unsupported line: {line}")


def _get_photoLineWV_fisspy(line: str, wvmin: float, wvmax: float):
    ll = line.lower()
    if ll == "ca":
        wvp, dwv = (8536.165, 0.25)
        if wvp > wvmin + 2 * dwv and wvp < wvmax - 2 * dwv:
            return (wvp, dwv)
        return (8548.079 * (1.0 + -0.62 / 300000.0), 0.25)
    raise ValueError(f"Unsupported line: {line}")


def _smooth_profiles_np(profiles: np.ndarray) -> np.ndarray:
    prof = np.asarray(profiles, dtype=np.float64)
    if prof.shape[-1] >= 7 and savgol_filter is not None:
        return savgol_filter(prof, window_length=7, polyorder=2, axis=-1)
    k = min(7, prof.shape[-1])
    if k <= 1:
        return prof.copy()
    if k % 2 == 0:
        k -= 1
    if k <= 1:
        return prof.copy()
    pad = k // 2
    kernel = np.ones(k, dtype=np.float64) / k
    prof_pad = np.pad(prof, ((0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(prof, dtype=np.float64)
    for i in range(prof.shape[0]):
        out[i] = np.convolve(prof_pad[i], kernel, mode="valid")
    return out


def _linecenter_polyfit_nd_np(
    wv: np.ndarray, prof: np.ndarray, nd: int = 2
) -> np.ndarray:
    wv = np.asarray(wv, dtype=np.float64)
    prof = np.asarray(prof, dtype=np.float64)
    N, M = prof.shape
    need = 2 * nd + 1
    if M < need:
        return wv[np.argmin(prof, axis=1)]
    s = np.argmin(prof[:, nd : M - nd], axis=1) + nd
    offsets = np.arange(-nd, nd + 1, dtype=np.int64)
    idx = s[:, None] + offsets[None, :]
    y = prof[np.arange(N)[:, None], idx]
    dx = float(np.median(np.diff(wv)))
    t = offsets.astype(np.float64) * dx
    A = np.stack([t * t, t, np.ones_like(t)], axis=1)
    pinv = np.linalg.pinv(A)
    coeff = y @ pinv.T
    a = coeff[:, 0]
    b = coeff[:, 1]
    xmid = wv[s]
    center = xmid.copy()
    good = np.abs(a) > 1e-14
    center[good] = xmid[good] - b[good] / (2.0 * a[good])
    return center


def estimate_vp_fisspy_from_profiles(
    wavelengths_A, profiles, line="ca", invert_for_current_forward=True
):
    """Estimate photospheric velocities from full observed spectra, in km/s."""
    wv = np.asarray(wavelengths_A, dtype=np.float64).reshape(-1)
    prof = np.asarray(profiles, dtype=np.float64)
    if prof.ndim != 2:
        raise ValueError(f"profiles must be 2D (Npix, Nλ), got {prof.shape}")
    if prof.shape[1] != wv.size:
        raise ValueError(
            f"profiles.shape[1]={prof.shape[1]} != len(wavelengths)={wv.size}"
        )
    wvline = _get_centerWV_local(line)
    wvp, dwv = _get_photoLineWV_fisspy(line, float(wv.min()), float(wv.max()))
    pline = np.abs(wv - wvp) <= 2.0 * dwv
    if pline.sum() < 5:
        raise RuntimeError(
            f"Not enough wavelength samples around the FISS-Py photospheric reference line. line={line}, ref={wvp:.6f} Å, selected={int(pline.sum())}"
        )
    profav = _smooth_profiles_np(prof)
    sub_wv_abs = wv[pline]
    sub_wv_rel = sub_wv_abs - wvp
    sub_prof = profav[:, pline]
    dwc = _linecenter_polyfit_nd_np(sub_wv_rel, sub_prof, nd=2)
    vp_fisspy = dwc / wvp * C_KMS
    if invert_for_current_forward:
        vp_used = -vp_fisspy
    else:
        vp_used = vp_fisspy
    meta = {
        "wvline": float(wvline),
        "wvp": float(wvp),
        "dwv": float(dwv),
        "invert_for_current_forward": bool(invert_for_current_forward),
    }
    return (vp_used.astype(np.float32), meta)


def overwrite_vp_with_fisspy(
    pred_params: torch.Tensor, vp_values: torch.Tensor
) -> torch.Tensor:
    out = pred_params.clone()
    out[:, 4] = vp_values.to(device=pred_params.device, dtype=pred_params.dtype)
    return out
