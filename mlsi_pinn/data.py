"""Read corrected spectra and explicit wavelength grids from FITS files."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

PARAMETER_NAMES = (
    "log_Sp",
    "log_S2",
    "log_S1",
    "log_S0",
    "vp",
    "v_upper",
    "v_lower",
    "log_wp",
    "log_w_upper",
    "log_w_lower",
    "log_ap",
    "log_eta",
    "tau_total",
)
PARAMETER_UNITS = (
    "log10(I/Iref)",
    "log10(I/Iref)",
    "log10(I/Iref)",
    "log10(I/Iref)",
    "km/s",
    "km/s",
    "km/s",
    "log10(Angstrom)",
    "log10(Angstrom)",
    "log10(Angstrom)",
    "log10(dimensionless)",
    "log10(dimensionless)",
    "dimensionless",
)
FISSPY_ORDER = (4, 5, 12, 13, 0, 9, 8, 2, 11, 10, 3, 1)


@dataclass
class Spectra:
    intensity: np.ndarray
    wavelength_A: np.ndarray
    reference: np.ndarray
    header: fits.Header

    @property
    def profiles(self):
        return self.intensity.reshape(-1, self.intensity.shape[-1])


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(intensity, wavelength_A, reference=None):
    if intensity.ndim not in (2, 3) or min(intensity.shape) < 1:
        raise ValueError(
            "Spectra must have shape (n, wavelength) or (row, column, wavelength)."
        )
    if wavelength_A.ndim != 1 or intensity.shape[-1] != wavelength_A.size:
        raise ValueError(
            "WAVELENGTH must be a vector matching the last intensity axis."
        )
    if not np.isfinite(intensity).all() or not np.isfinite(wavelength_A).all():
        raise ValueError("Spectra and wavelengths must contain only finite values.")
    if wavelength_A.size < 7 or not np.all(np.diff(wavelength_A) > 0):
        raise ValueError(
            "Provide at least seven wavelengths in increasing angstrom order."
        )
    if not (8500 < wavelength_A.min() < wavelength_A.max() < 8600):
        raise ValueError(
            "This example expects absolute Ca II wavelengths in angstroms."
        )
    if reference is not None:
        if reference.shape != intensity.shape[:-1] + (13,):
            raise ValueError(
                "REFERENCE must match the spatial shape with 13 parameters last."
            )
        if not np.isfinite(reference).all():
            raise ValueError("Reference parameters must contain only finite values.")


def read_spectra(path, require_reference=False):
    with fits.open(path, memmap=False, checksum=True) as hdus:
        intensity = np.asarray(hdus[0].data, dtype=np.float32).copy()
        header = hdus[0].header.copy()
        if "WAVELENGTH" not in hdus:
            raise ValueError(
                "Missing WAVELENGTH extension. Convert the file with prepare_data.py."
            )
        wavelength = np.asarray(hdus["WAVELENGTH"].data, dtype=np.float32).copy()
        if hdus["WAVELENGTH"].header.get("BUNIT") != "Angstrom":
            raise ValueError("WAVELENGTH BUNIT must be Angstrom.")
        reference = None
        if "REFERENCE" in hdus:
            reference = np.asarray(hdus["REFERENCE"].data, dtype=np.float32).copy()
            names = tuple(
                hdus["REFERENCE"].header.get("PAR%02d" % i) for i in range(13)
            )
            if names != PARAMETER_NAMES:
                raise ValueError(
                    "REFERENCE parameter order differs from the package schema."
                )
    validate(intensity, wavelength, reference)
    if require_reference and reference is None:
        raise ValueError("Two-stage training requires the REFERENCE extension.")
    return Spectra(intensity, wavelength, reference, header)


def parameter_hdu(parameters, name="PARAMETERS"):
    hdu = fits.ImageHDU(np.asarray(parameters, dtype=np.float32), name=name)
    for i, (param, unit) in enumerate(zip(PARAMETER_NAMES, PARAMETER_UNITS)):
        hdu.header["PAR%02d" % i] = param
        hdu.header["UNIT%02d" % i] = unit
    hdu.header["VCONVENT"] = (
        "FISSPy reference, unchanged"
        if name == "REFERENCE"
        else "model: positive = blueshift"
    )
    return hdu


def wavelength_hdu(wavelength_A):
    hdu = fits.ImageHDU(np.asarray(wavelength_A, dtype=np.float32), name="WAVELENGTH")
    hdu.header["BUNIT"] = "Angstrom"
    return hdu


def write_spectra(
    path, intensity, wavelength_A, reference=None, header=None, overwrite=False
):
    intensity = np.asarray(intensity, dtype=np.float32)
    wavelength_A = np.asarray(wavelength_A, dtype=np.float32)
    reference = None if reference is None else np.asarray(reference, dtype=np.float32)
    validate(intensity, wavelength_A, reference)
    primary = fits.PrimaryHDU(intensity, header=header)
    primary.header["DATATYPE"] = "CORRECTED_SPECTRA"
    primary.header["BUNIT"] = "normalized intensity"
    primary.header["SCHEMA"] = "MLSI-PINN Ca 1"
    hdus = [primary, wavelength_hdu(wavelength_A)]
    if reference is not None:
        hdus.append(parameter_hdu(reference, "REFERENCE"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite, checksum=True)


def read_fisspy_parameters(path, spatial_shape):
    """Decode a conventional MLSI parameter FITS cube with parameters on axis 0."""
    with fits.open(path, memmap=False) as hdus:
        data = np.asarray(hdus[0].data, dtype=np.float32).copy()
        header = hdus[0].header
        if (
            data.ndim != 3
            or data.shape[0] < 15
            or data.shape[1:] != tuple(spatial_shape)
        ):
            raise ValueError("Expected a (>=15, row, column) MLSI parameter cube.")
        scaled = any("SCALE%02d" % i in header for i in range(data.shape[0]))
        if scaled:
            for i in range(data.shape[0]):
                if "SCALE%02d" % i not in header:
                    raise ValueError("Incomplete SCALEnn parameter metadata.")
                data[i] *= header["SCALE%02d" % i]
        elif np.issubdtype(hdus[0].data.dtype, np.integer):
            raise ValueError("Integer MLSI parameters require SCALEnn metadata.")
    result = np.empty(tuple(spatial_shape) + (13,), dtype=np.float32)
    for target, source in enumerate(FISSPY_ORDER):
        result[..., target] = data[source]
    result[..., 12] = (
        10.0 ** data[6].astype(np.float64) + 10.0 ** data[7].astype(np.float64)
    ).astype(np.float32)
    return result
