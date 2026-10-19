"""Package corrected spectra with an explicit wavelength vector and optional MLSI targets."""

import argparse
import numpy as np
from astropy.io import fits
from mlsi_pinn.data import read_fisspy_parameters, write_spectra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectra", required=True, help="Corrected FITS cube, wavelength last."
    )
    parser.add_argument(
        "--wavelength",
        required=True,
        help="NumPy .npy vector of absolute angstrom wavelengths.",
    )
    parser.add_argument(
        "--parameters", help="Conventional MLSI parameter FITS, parameter axis first."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with fits.open(args.spectra, memmap=False) as hdus:
        intensity = np.asarray(hdus[0].data, dtype=np.float32).copy()
        header = fits.Header()
        for key in ("DATE-OBS", "FILEORIG"):
            if key in hdus[0].header:
                header[key] = hdus[0].header[key]
    reference = None
    if args.parameters:
        reference = read_fisspy_parameters(args.parameters, intensity.shape[:-1])
    wavelength = np.load(args.wavelength, allow_pickle=False)
    write_spectra(args.output, intensity, wavelength, reference, header, args.overwrite)
    print("Saved " + args.output)


if __name__ == "__main__":
    main()
