"""Inspect a prepared Ca II dataset without training."""

import argparse
import json
from mlsi_pinn.data import read_spectra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="examples/data/train.fits.gz")
    args = parser.parse_args()
    data = read_spectra(args.path)
    print(
        json.dumps(
            dict(
                shape=list(data.intensity.shape),
                wavelength_A=[
                    float(data.wavelength_A[0]),
                    float(data.wavelength_A[-1]),
                ],
                reference_parameters=data.reference is not None,
                date_obs=data.header.get("DATE-OBS"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
