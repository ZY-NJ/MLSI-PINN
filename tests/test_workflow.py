import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from mlsi_pinn.data import PARAMETER_NAMES, read_spectra, write_spectra
from mlsi_pinn.inference import infer
from mlsi_pinn.model import MLSIPINN
from mlsi_pinn.physics import mlsi_forward_torch_batch
from mlsi_pinn.training import DEFAULT_CONFIG, freeze_for_stage2, train

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.data = read_spectra(ROOT / "examples/data/train.fits.gz", True)

    def test_data_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectra.fits.gz"
            write_spectra(
                path,
                self.data.intensity[:2],
                self.data.wavelength_A,
                self.data.reference[:2],
            )
            actual = read_spectra(path, True)
            np.testing.assert_array_equal(actual.intensity, self.data.intensity[:2])
            np.testing.assert_array_equal(actual.reference, self.data.reference[:2])
            np.testing.assert_array_equal(actual.wavelength_A, self.data.wavelength_A)
        with self.assertRaisesRegex(ValueError, "REFERENCE"):
            read_spectra(ROOT / "examples/data/inference.fits.gz", True)

    def test_forward_gradient_and_frozen_rows(self):
        torch.manual_seed(42)
        model = MLSIPINN(len(self.data.wavelength_A))
        observed = torch.tensor(self.data.profiles[:8])
        wave = torch.tensor(self.data.wavelength_A)
        before = {key: value.clone() for key, value in model.named_parameters()}
        hooks = freeze_for_stage2(model, DEFAULT_CONFIG["stage2_trainable_rows"])
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.0006
        )
        prediction = model(observed)
        reconstructed = mlsi_forward_torch_batch(wave, prediction)
        (reconstructed[:, 112:360] - observed[:, 112:360]).square().mean().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        frozen = [0, 1, 4, 7, 10, 11, 12]
        for name, parameter in model.named_parameters():
            if name.startswith("encoder.10."):
                self.assertTrue(torch.equal(parameter[frozen], before[name][frozen]))
            else:
                self.assertTrue(torch.equal(parameter, before[name]))
        self.assertFalse(
            torch.equal(model.encoder[-1].weight, before["encoder.10.weight"])
        )
        for hook in hooks:
            hook.remove()

    def test_cpu_training_and_independent_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            training = base / "train.fits.gz"
            write_spectra(
                training,
                self.data.profiles[:32],
                self.data.wavelength_A,
                self.data.reference.reshape(-1, 13)[:32],
            )
            result = train(
                training,
                base / "run",
                {"stage1_epochs": 1, "stage2_epochs": 1, "batch_size": 16},
                device="cpu",
            )
            self.assertEqual(result["n_spectra"], 32)
            independent = read_spectra(ROOT / "examples/data/inference.fits.gz")
            test_input = base / "inference.fits.gz"
            write_spectra(
                test_input, independent.profiles[:9], independent.wavelength_A
            )
            parameters = infer(
                test_input,
                base / "run/model.pt",
                base / "result.fits.gz",
                device="cpu",
                batch_size=4,
                synthesize=True,
            )
            self.assertEqual(parameters.shape, (9, 13))
            with fits.open(base / "result.fits.gz", checksum=True) as hdus:
                self.assertEqual(hdus["SYNTHESIS"].data.shape, (9, 502))
                self.assertTrue(np.isfinite(hdus["SYNTHESIS"].data).all())
                for i, name in enumerate(PARAMETER_NAMES):
                    self.assertEqual(hdus["PARAMETERS"].header["PAR%02d" % i], name)
                for hdu in hdus:
                    self.assertEqual(hdu.verify_checksum(), 1)
            write_spectra(
                base / "wrong_grid.fits",
                independent.profiles[:9],
                independent.wavelength_A + np.float32(0.01),
            )
            with self.assertRaisesRegex(ValueError, "wavelength grid"):
                infer(
                    base / "wrong_grid.fits", base / "run/model.pt", base / "wrong.fits"
                )


if __name__ == "__main__":
    unittest.main()
