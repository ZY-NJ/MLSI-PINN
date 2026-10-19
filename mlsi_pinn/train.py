"""Train the Ca II example in two stages."""

import argparse
import json
from mlsi_pinn.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/train.fits.gz")
    parser.add_argument("--config", default="configs/ca_example.json")
    parser.add_argument("--output", default="runs/ca_example")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--stage1-epochs", type=int)
    parser.add_argument("--stage2-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = json.load(stream)
    for key in ("stage1_epochs", "stage2_epochs", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    result = train(args.data, args.output, config, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
