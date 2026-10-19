# MLSI-PINN

A runnable Ca II 8542 example of multilayer spectral inversion with a physics-informed neural network. The package includes corrected FISS spectra, conventional MLSI reference parameters for training, a separate inference raster, and a trained example checkpoint.

## Quick start

Run these commands from the repository directory. Python 3.8 or later is required. The tested dependency versions are also recorded in `requirements-tested.txt`. A CUDA GPU accelerates training, but the example also runs on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python read_data.py examples/data/train.fits.gz
python infer.py --synthesize
```

The inference command uses `examples/ca_example.pt` and writes `outputs/inference.fits.gz`. Omit `--synthesize` to retrieve parameters only. On Windows, activate the environment with `.venv\Scripts\activate`.

To train a new model and use it on the independent inference raster:

```bash
python train.py --device auto --output runs/ca_example
python infer.py --checkpoint runs/ca_example/model.pt --output outputs/retrained.fits.gz --synthesize
```

The default schedule is 50 epochs in Stage 1 and 20 epochs in Stage 2. For a short execution check, use a separate output directory:

```bash
python train.py --stage1-epochs 1 --stage2-epochs 1 --output runs/smoke
python -m unittest discover -s tests -v
```

A one-epoch run checks execution only. It is not a trained scientific result. Existing training directories are never overwritten. Inference outputs require `--overwrite` to replace an existing file.

## Contents

| File | Purpose |
|---|---|
| `read_data.py` | Inspect prepared spectra and their wavelength grid |
| `prepare_data.py` | Package corrected FITS spectra and optional MLSI targets |
| `train.py` | Train both stages and save the selected checkpoints |
| `infer.py` | Estimate parameters, optionally reconstruct spectra |
| `mlsi_pinn/model.py` | Network and bounded parameter mapping |
| `mlsi_pinn/physics.py` | Differentiable three-layer forward calculation |
| `mlsi_pinn/velocity.py` | Photospheric velocity estimate from observed spectra |
| `mlsi_pinn/data.py` | FITS schema, parameter scaling and data validation |
| `configs/ca_example.json` | Complete training configuration |
| `examples/data/` | One training crop and one separate inference crop |
| `examples/ca_example.pt` | Checkpoint trained on the included training crop |
| `examples/training_history.csv` | Per-epoch spectral, parameter and total losses |

## Data and preprocessing

Each example is a 64 × 64 spatial crop with all 502 wavelength samples retained. The training raster is from 2023-08-14 at 17:59:02.489 and the inference raster is from the same day at 17:25:19.159. The latter is not used for training or checkpoint selection. `examples/data/manifest.json` records source filenames, crops and SHA-256 hashes.

The input spectra have already been normalized and corrected with FISSPy `corAll`. The reader performs no additional normalization, spatial transposition or wavelength reversal. The original corrected files lacked wavelength calibration cards. The bundled wavelength vector therefore preserves the float32 grid reconstructed by the retained training loader. It is recorded explicitly in each example, rather than silently assumed for new data. See `examples/data/README.md` for the grid construction.

For your own observations, first obtain consistently normalized and corrected spectra with their actual wavelength calibration. `prepare_data.py` converts those products to the example format:

```bash
python prepare_data.py --spectra corrected.fits --wavelength wavelength_A.npy \
    --parameters mlsi_parameters.fits --output training.fits.gz
python prepare_data.py --spectra other_corrected.fits --wavelength wavelength_A.npy \
    --output inference.fits.gz
```

Spectra must have the wavelength axis last and wavelengths must increase in absolute angstroms. Conventional MLSI parameter files use the parameter axis first, with `SCALEnn` cards for integer encoding. Inference data do not need reference parameters. Raw compressed FISS files are not direct inputs to these commands.

Training and inference must use the same wavelength grid, normalization and correction convention. An incompatible inference grid is rejected. The full observed wavelength range must include the photospheric reference line needed to estimate vₚ. The supplied setup is intended for same-day observations with consistent preprocessing.

## Training

The network has hidden widths 256, 128 and 64, followed by 13 outputs and the retained physical parameter mapping. Dropout is zero in this example. Both stages use Adam and seed 42.

- **Stage 1:** update all network weights using 10 times the spectral MSE over samples `[112:360)`.
- **Stage 2:** update final-layer rows `[2, 3, 5, 6, 8, 9]` using 10 times the spectral MSE over `[210:238)`, plus the parameter MSE for the two chromospheric velocities and log widths. Other weights are frozen. As in the retained implementation, BatchNorm running statistics continue to update in training mode.

Slice endpoints are zero-based and exclusive. The windows are specific to the supplied wavelength grid. The exact learning rates and other settings are in `configs/ca_example.json`.

The lowest training-objective checkpoint is selected in each stage. Stage 2 also considers its initial checkpoint before updates. The best Stage 1 state initializes Stage 2. Training writes `stage1.pt`, `model.pt`, `history.csv` and `summary.json`. No inference-raster metric is used for model selection. The checkpoint includes the wavelength vector, configuration, parameter order and training-data hash. The seed fixes this example's random choices, but bitwise reproducibility across hardware and library versions is not guaranteed.

## Parameters and outputs

FITS inputs contain corrected intensities in the primary HDU, a `WAVELENGTH` extension in angstroms, and an optional `REFERENCE` extension with 13 parameters last. Inference outputs contain `PARAMETERS`, `WAVELENGTH` and, when requested, `SYNTHESIS`. Each parameter's name and units are stored in the header.

| Index | Output name | Meaning and units |
|---|---|---|
| 0–3 | `log_Sp`, `log_S2`, `log_S1`, `log_S0` | Base-10 log source functions, normalized intensity |
| 4 | `vp` | Photospheric velocity estimated separately, km/s |
| 5–6 | `v_upper`, `v_lower` | Chromospheric velocities, km/s |
| 7–9 | `log_wp`, `log_w_upper`, `log_w_lower` | Base-10 log Doppler widths in angstroms |
| 10 | `log_ap` | Base-10 log photospheric damping parameter |
| 11 | `log_eta` | Base-10 log photospheric line-to-continuum opacity ratio |
| 12 | `tau_total` | Fixed total chromospheric optical thickness of 10 |

The upper/lower chromospheric outputs correspond to v₁/w₁ and v₂/w₂ in the manuscript. The retained model uses positive velocity for a blueward shift. The vₚ estimate is sign-adjusted for this convention. The conventional MLSI reference targets are decoded and copied without a new sign transformation, preserving the training implementation. Reference values are comparison targets, not independently known physical truth.

Widths are stored logarithmically. Convert with `width_A = 10**log_width` or `width_pm = 100 * 10**log_width`. The forward model performs this conversion internally. Photospheric width is tied to `log_Sp`, vₚ is replaced by its separate estimate, and `tau_total` is fixed. They are not three independently fitted network outputs.

The retained mapping bounds the four log source functions, both chromospheric velocities, the chromospheric log widths and `log_eta`. It also imposes `log_S0 <= log_S1 - 0.05`. `log_ap` remains unbounded, as in the source implementation. The exact mapping is readable in `model.py`.

## Scope and attribution

Software is provided under the BSD 2-Clause License. See `LICENSE` and `THIRD_PARTY_NOTICES.txt`. Example observations retain their original data rights.

This is the Ca II configuration, with compact demonstration data. The included checkpoint was trained on the included crop. It is not the full-raster checkpoint used for the manuscript figures. The differentiable forward model retains the original Voigt approximation, and should not be described as numerically identical to FISSPy's conventional forward solver. Parameter bounds and reference supervision do not establish a unique physical inversion.

The data correction, conventional reference inversions and photospheric reference-line procedure use or follow [FISSPy](https://pypi.org/project/fisspy/). FISSPy is not needed to run the prepared examples. The retained source was checked against FISSPy 1.0.0. Third-party attribution is in `THIRD_PARTY_NOTICES.txt`, and data provenance is in `examples/data/README.md`.
