"""Retrieve Ca II parameters from a corrected raster."""

import argparse
from mlsi_pinn.inference import infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/inference.fits.gz")
    parser.add_argument("--checkpoint", default="examples/ca_example.pt")
    parser.add_argument("--output", default="outputs/inference.fits.gz")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Also write the reconstructed spectrum.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    parameters = infer(
        args.data,
        args.checkpoint,
        args.output,
        args.device,
        args.batch_size,
        args.synthesize,
        args.overwrite,
    )
    print("Saved %s, parameter shape %s" % (args.output, parameters.shape))


if __name__ == "__main__":
    main()
