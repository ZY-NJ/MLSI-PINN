"""Parameter retrieval, with optional spectral synthesis."""

from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from .data import (
    PARAMETER_NAMES,
    file_sha256,
    parameter_hdu,
    read_spectra,
    wavelength_hdu,
)
from .model import MLSIPINN
from .physics import mlsi_forward_torch_batch
from .training import resolve_device
from .velocity import estimate_vp_fisspy_from_profiles, overwrite_vp_with_fisspy


def infer(
    data_path,
    checkpoint_path,
    output_path,
    device="auto",
    batch_size=256,
    synthesize=False,
    overwrite=False,
):
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if Path(output_path).exists() and not overwrite:
        raise FileExistsError(output_path)
    data = read_spectra(data_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint["parameter_names"] != list(
        PARAMETER_NAMES
    ):
        raise ValueError("Unsupported checkpoint schema or parameter order.")
    expected = checkpoint["wavelength_A"].numpy()
    if data.wavelength_A.shape != expected.shape or not np.array_equal(
        data.wavelength_A, expected
    ):
        raise ValueError(
            "Input wavelength grid differs from training. Prepare a matching grid explicitly."
        )
    device = resolve_device(device)
    torch.set_num_threads(4)
    model = MLSIPINN(len(expected), checkpoint["config"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    vp, _ = estimate_vp_fisspy_from_profiles(data.wavelength_A, data.profiles)
    wavelength = torch.tensor(data.wavelength_A, device=device)
    parameters = np.empty((len(vp), 13), dtype=np.float32)
    spectra = np.empty_like(data.profiles) if synthesize else None
    with torch.inference_mode():
        for start in range(0, len(vp), batch_size):
            stop = min(start + batch_size, len(vp))
            observed = torch.tensor(data.profiles[start:stop], device=device)
            batch_vp = torch.tensor(vp[start:stop], device=device)
            predicted = overwrite_vp_with_fisspy(model(observed), batch_vp)
            parameters[start:stop] = predicted.cpu().numpy()
            if synthesize:
                spectra[start:stop] = (
                    mlsi_forward_torch_batch(wavelength, predicted).cpu().numpy()
                )
    if not np.isfinite(parameters).all() or (
        synthesize and not np.isfinite(spectra).all()
    ):
        raise RuntimeError("Nonfinite inference result.")
    primary = fits.PrimaryHDU()
    primary.header["SCHEMA"] = "MLSI-PINN Ca 1"
    primary.header["MODEL"] = Path(checkpoint_path).name
    primary.header["MODSHA"] = file_sha256(checkpoint_path)
    primary.header["DATASHA"] = file_sha256(data_path)
    if "DATE-OBS" in data.header:
        primary.header["DATE-OBS"] = data.header["DATE-OBS"]
    parameters = parameters.reshape(data.intensity.shape[:-1] + (13,))
    hdus = [primary, parameter_hdu(parameters), wavelength_hdu(data.wavelength_A)]
    if synthesize:
        synthesis_hdu = fits.ImageHDU(
            spectra.reshape(data.intensity.shape), name="SYNTHESIS"
        )
        synthesis_hdu.header["BUNIT"] = "normalized intensity"
        hdus.append(synthesis_hdu)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(output_path, overwrite=overwrite, checksum=True)
    return parameters
