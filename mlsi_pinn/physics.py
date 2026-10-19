"""Differentiable Ca II three-layer synthesis retained from the research model.

Wavelength inputs are absolute angstroms. The photospheric and lower-layer
profiles use the retained differentiable Voigt approximation.
"""

import torch

LINE_CENTERS_A = {"ca": 8542.091}


def _relative_wavelength_A(lam_A: torch.Tensor, line="ca"):
    lam0 = LINE_CENTERS_A[line.lower()]
    return lam_A - lam0


def _expand_col(x):
    return x.unsqueeze(-1)


def _voigt_H_approx(
    a: torch.Tensor, u: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    a = torch.clamp(a, min=0.0)
    u2 = u * u
    a2 = a * a
    G = torch.exp(-u2)
    L = (a2 + eps) / (u2 + a2 + eps)
    t = a / (a + 0.5 + eps) + u2 / (u2 + 4.0 + eps) * (a / (a + 0.2 + eps))
    eta = torch.clamp(t, 0.0, 1.0)
    H = (1.0 - eta) * G + eta * L
    return H


def _absP_Gauss_broadcast(wv_A, center_A, width_A, line="ca"):
    eps = torch.finfo(wv_A.dtype).eps
    width_A = torch.clamp(width_A, min=eps)
    shape_prefix = center_A.shape[:-2]
    wv_b = wv_A.view(*[1] * len(shape_prefix), 1, -1)
    u = (wv_b - center_A) / width_A
    dwv = (
        torch.tensor(
            [0.0857, 0.1426, 0.1696, 0.1952, 0.2433, 0.2871],
            dtype=wv_A.dtype,
            device=wv_A.device,
        )
        - 0.0857
    )
    dA = 10 ** (
        torch.tensor(
            [6.33, 4.15, 3.47, 4.66, 1.94, 3.61], dtype=wv_A.dtype, device=wv_A.device
        )
        - 6.33
    )
    Phi = torch.exp(-((u - dwv[0] / width_A) ** 2)) * dA[0]
    return Phi


def _absP_Voigt_broadcast(wv_A, center_A, width_A, a, line="ca"):
    eps = torch.finfo(wv_A.dtype).eps
    width_A = torch.clamp(width_A, min=eps)
    a = torch.clamp(a, min=0.0)
    shape_prefix = center_A.shape[:-2]
    wv_b = wv_A.view(*[1] * len(shape_prefix), 1, -1)
    u = (wv_b - center_A) / width_A
    dwv = (
        torch.tensor(
            [0.0857, 0.1426, 0.1696, 0.1952, 0.2433, 0.2871],
            dtype=wv_A.dtype,
            device=wv_A.device,
        )
        - 0.0857
    )
    dA = 10 ** (
        torch.tensor(
            [6.33, 4.15, 3.47, 4.66, 1.94, 3.61], dtype=wv_A.dtype, device=wv_A.device
        )
        - 6.33
    )
    u0 = u - dwv[0] / width_A
    H = _voigt_H_approx(a, u0, eps=float(eps)) * dA[0]
    H0 = _voigt_H_approx(a, torch.zeros_like(u0[..., :1]), eps=float(eps))
    Phi = H / torch.clamp(H0, min=eps)
    return Phi


def _mlsi_forward_core(wv1_A, par15, line="ca"):
    """Return the emergent, lower-layer and photospheric spectra for 15 parameters."""
    device, dtype = (par15.device, par15.dtype)
    B = par15.shape[0]
    wv1 = wv1_A.to(device=device, dtype=dtype)
    wvline_A = torch.tensor(8542.09, device=device, dtype=dtype)
    c_kms = torch.tensor(299792.458, device=device, dtype=dtype)
    vp = par15[:, 0]
    wvp = _expand_col(-vp / c_kms * wvline_A)
    eta = _expand_col(torch.pow(10.0, par15[:, 1]))
    wp = _expand_col(torch.pow(10.0, par15[:, 2]))
    ap = _expand_col(torch.pow(10.0, par15[:, 3]))
    Sp = _expand_col(torch.pow(10.0, par15[:, 4]))
    S2 = _expand_col(torch.pow(10.0, par15[:, 5]))
    tau2 = _expand_col(torch.pow(10.0, par15[:, 6]))
    tau1 = _expand_col(torch.pow(10.0, par15[:, 7]))
    v1 = par15[:, 8]
    wv01 = _expand_col(-v1 / c_kms * wvline_A)
    v0 = par15[:, 9]
    wv00 = _expand_col(-v0 / c_kms * wvline_A)
    w1 = _expand_col(torch.pow(10.0, par15[:, 10]))
    w0 = _expand_col(torch.pow(10.0, par15[:, 11]))
    S1 = _expand_col(torch.pow(10.0, par15[:, 12]))
    S0 = _expand_col(torch.pow(10.0, par15[:, 13]))
    wg = _expand_col(par15[:, 14])
    wv02 = wvp
    w2 = wp
    rlamb = eta * _absP_Voigt_broadcast(wv1, wv02, wp, ap).squeeze(0) + 1.0
    I2 = S2 + (Sp - S2) / rlamb
    xvals = torch.tensor([-0.774597, 0.0, 0.774597], device=device, dtype=dtype)
    wghts = torch.tensor([0.55556, 0.888889, 0.555556], device=device, dtype=dtype)
    x_outer = (xvals + 1.0) / 2.0
    x_outer = x_outer.view(3, 1, 1)
    x_inner = (xvals.view(3, 1, 1, 1) + 1.0) * x_outer / 2.0

    def B1(t):
        return t.view(1, -1, 1)

    wvcenter_lc = B1(wv01) + (B1(wv02) - B1(wv01)) * x_outer
    width_lc = B1(w1) + (B1(w2) - B1(w1)) * x_outer
    a_lc = B1(wg) * x_outer / torch.clamp(width_lc, min=torch.finfo(dtype).eps)
    V_outer = _absP_Voigt_broadcast(wv1, wvcenter_lc, width_lc, a_lc)
    dummy_lc = torch.sum(wghts.view(3, 1, 1) * V_outer, dim=0)
    taulamb_lc = tau2 / 2.0 * dummy_lc
    A0 = S0
    A1 = -1.5 * S0 + 2.0 * S1 - 0.5 * S2
    A2 = 0.5 * S0 - 1.0 * S1 + 0.5 * S2
    wvcenter_inner = B1(wv01) + (B1(wv02) - B1(wv01)) * x_inner
    width_inner = B1(w1) + (B1(w2) - B1(w1)) * x_inner
    a_inner = B1(wg) * x_inner / torch.clamp(width_inner, min=torch.finfo(dtype).eps)
    V_inner = _absP_Voigt_broadcast(wv1, wvcenter_inner, width_inner, a_inner)
    dummy_inner = torch.sum(wghts.view(3, 1, 1, 1) * V_inner, dim=0)
    tau2_outer = x_outer.view(3, 1, 1) * (tau2.view(1, B, 1) / 2.0)
    tlamb_inner = tau2_outer * dummy_inner

    def _S_from_x(x, A0v, A1v, A2v, x0):
        return A0v + A1v * (x + x0) + A2v * (x + x0) ** 2

    S_outer_lc = _S_from_x(
        x_outer, A0.view(1, B, 1), A1.view(1, B, 1), A2.view(1, B, 1), x0=1.0
    )
    integral_lc = torch.sum(
        wghts.view(3, 1, 1) * S_outer_lc * torch.exp(-tlamb_inner) * V_outer, dim=0
    )
    I1 = I2 * torch.exp(-taulamb_lc) + tau2 / 2.0 * integral_lc
    wvcenter_uc = B1(wv00) + (B1(wv01) - B1(wv00)) * x_outer
    width_uc = B1(w0) + (B1(w1) - B1(w0)) * x_outer
    G_outer = _absP_Gauss_broadcast(wv1, wvcenter_uc, width_uc)
    dummy_uc = torch.sum(wghts.view(3, 1, 1) * G_outer, dim=0)
    taulamb_uc = tau1 / 2.0 * dummy_uc
    wvcenter_inner_uc = B1(wv00) + (B1(wv01) - B1(wv00)) * x_inner
    width_inner_uc = B1(w0) + (B1(w1) - B1(w0)) * x_inner
    G_inner = _absP_Gauss_broadcast(wv1, wvcenter_inner_uc, width_inner_uc)
    dummy_inner_uc = torch.sum(wghts.view(3, 1, 1, 1) * G_inner, dim=0)
    tau1_outer = x_outer.view(3, 1, 1) * (tau1.view(1, B, 1) / 2.0)
    tlamb_inner_uc = tau1_outer * dummy_inner_uc
    S_outer_uc = _S_from_x(
        x_outer, A0.view(1, B, 1), A1.view(1, B, 1), A2.view(1, B, 1), x0=0.0
    )
    integral_uc = torch.sum(
        wghts.view(3, 1, 1) * S_outer_uc * torch.exp(-tlamb_inner_uc) * G_outer, dim=0
    )
    I0 = I1 * torch.exp(-taulamb_uc) + tau1 / 2.0 * integral_uc
    return (I0, I1, I2)


def mlsi_forward_torch_batch(wavelength_A, params13, line="ca"):
    """Return the emergent spectrum. Wavelengths are absolute angstroms."""
    if line.lower() != "ca":
        raise ValueError("This package provides the Ca II configuration only.")
    device, dtype = (params13.device, params13.dtype)
    log_Sp = params13[:, 0]
    log_S2 = params13[:, 1]
    log_S1 = params13[:, 2]
    log_S0 = params13[:, 3]
    vp = params13[:, 4]
    v0 = params13[:, 5]
    v1 = params13[:, 6]
    log_wp = params13[:, 7]
    log_w0 = params13[:, 8]
    log_w1 = params13[:, 9]
    log_ap = params13[:, 10]
    log_eta = params13[:, 11]
    tau0 = params13[:, 12]
    wp = _expand_col(torch.pow(10.0, log_wp))
    tau2 = tau0 * 0.5
    tau1 = tau0 * 0.5
    wg = (wp * 0.001).clamp_min(1e-06)
    wg = wg[:, 0]
    par15 = torch.stack(
        [
            vp,
            log_eta,
            log_wp,
            log_ap,
            log_Sp,
            log_S2,
            torch.log10(tau2.clamp_min(1e-06)),
            torch.log10(tau1.clamp_min(1e-06)),
            v1,
            v0,
            log_w1,
            log_w0,
            log_S1,
            log_S0,
            wg,
        ],
        dim=1,
    ).to(dtype=dtype, device=device)
    wv1_A = _relative_wavelength_A(wavelength_A, line=line)
    I0, _, _ = _mlsi_forward_core(wv1_A, par15, line=line)
    return I0
