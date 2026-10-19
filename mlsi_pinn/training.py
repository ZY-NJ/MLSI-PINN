"""Two-stage training for the retained Ca II model."""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import PARAMETER_NAMES, file_sha256, read_spectra
from .model import MLSIPINN
from .physics import mlsi_forward_torch_batch
from .velocity import estimate_vp_fisspy_from_profiles, overwrite_vp_with_fisspy

DEFAULT_CONFIG = {
    "seed": 42,
    "batch_size": 256,
    "dropout": 0.0,
    "stage1_epochs": 50,
    "stage2_epochs": 20,
    "stage1_lr": 0.0002,
    "stage2_lr": 0.0006,
    "stage1_weight_decay": 0.00001,
    "stage2_weight_decay": 0.0,
    "stage1_window": [112, 360],
    "stage2_window": [210, 238],
    "spectral_weight": 10.0,
    "parameter_weight": 1.0,
    "stage2_trainable_rows": [2, 3, 5, 6, 8, 9],
    "stage2_supervised_rows": [5, 6, 8, 9],
    "stage2_grad_clip": 1.0,
}


def resolve_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu.")
    return torch.device(name)


def batch_indices(indices, batch_size):
    """Merge a singleton final training batch into its preceding batch."""
    start = 0
    while start < len(indices):
        stop = min(start + batch_size, len(indices))
        if len(indices) - stop == 1:
            stop += 1
        yield indices[start:stop]
        start = stop


def freeze_for_stage2(model, rows):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = model.encoder[-1]
    hooks = []
    for parameter in (layer.weight, layer.bias):
        parameter.requires_grad_(True)
        mask = torch.zeros_like(parameter)
        mask[rows] = 1.0
        hooks.append(
            parameter.register_hook(lambda gradient, mask=mask: gradient * mask)
        )
    return hooks


def objective(parameters, observed, wavelength, target, stage, config):
    synthesized = mlsi_forward_torch_batch(wavelength, parameters)
    start, stop = config["stage%d_window" % stage]
    spectral = (synthesized[:, start:stop] - observed[:, start:stop]).square().mean()
    parameter = spectral.new_zeros(())
    if stage == 2:
        rows = config["stage2_supervised_rows"]
        parameter = (parameters[:, rows] - target[:, rows]).square().mean()
        parameter = parameter * config["parameter_weight"]
    return config["spectral_weight"] * spectral + parameter, spectral, parameter


def cpu_state(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def train(data_path, output_dir, config=None, device="auto"):
    settings = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(settings)
        if unknown:
            raise ValueError("Unknown configuration keys: %s" % sorted(unknown))
        settings.update(config)
    data = read_spectra(data_path, require_reference=True)
    if len(data.profiles) < 2 or settings["batch_size"] < 2:
        raise ValueError("BatchNorm training requires at least two spectra per batch.")
    if settings["stage2_weight_decay"] != 0:
        raise ValueError(
            "Stage 2 weight decay must be zero to preserve frozen output rows."
        )
    for stage in (1, 2):
        start, stop = settings["stage%d_window" % stage]
        if not (0 <= start < stop <= data.wavelength_A.size):
            raise ValueError("Loss window lies outside the wavelength grid.")
        if settings["stage%d_epochs" % stage] < 1:
            raise ValueError("Both training stages require at least one epoch.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            "Use an empty output directory to preserve existing runs."
        )
    device = resolve_device(device)
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data.profiles))
    rng.shuffle(indices)
    vp, vp_metadata = estimate_vp_fisspy_from_profiles(data.wavelength_A, data.profiles)
    observed = torch.from_numpy(data.profiles)
    target = torch.from_numpy(data.reference.reshape(-1, 13))
    velocities = torch.from_numpy(vp)
    wavelength = torch.tensor(data.wavelength_A, device=device)
    model = MLSIPINN(len(data.wavelength_A), settings["dropout"]).to(device)
    history = []
    selected = {}
    started = time.perf_counter()

    def batch_loss(index, stage):
        spectra = observed[index].to(device)
        parameters = overwrite_vp_with_fisspy(
            model(spectra), velocities[index].to(device)
        )
        return objective(
            parameters, spectra, wavelength, target[index].to(device), stage, settings
        )

    for stage in (1, 2):
        hooks = (
            freeze_for_stage2(model, settings["stage2_trainable_rows"])
            if stage == 2
            else []
        )
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=settings["stage%d_lr" % stage],
            weight_decay=settings["stage%d_weight_decay" % stage],
        )
        best_state, best_epoch, best_loss = cpu_state(model), 0, float("inf")
        if stage == 2:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for index in batch_indices(indices, settings["batch_size"]):
                    total += float(batch_loss(index, stage)[0]) * len(index)
            best_loss = total / len(indices)
        try:
            for epoch in range(1, settings["stage%d_epochs" % stage] + 1):
                tick = time.perf_counter()
                model.train()
                rng.shuffle(indices)
                sums = np.zeros(3, dtype=np.float64)
                for index in batch_indices(indices, settings["batch_size"]):
                    losses = batch_loss(index, stage)
                    if not torch.isfinite(losses[0]):
                        raise RuntimeError("Nonfinite training objective.")
                    optimizer.zero_grad(set_to_none=True)
                    losses[0].backward()
                    if stage == 2:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            settings["stage2_grad_clip"],
                        )
                    optimizer.step()
                    sums += np.array([float(value.detach()) for value in losses]) * len(
                        index
                    )
                means = sums / len(indices)
                if means[0] < best_loss:
                    best_state, best_epoch, best_loss = (
                        cpu_state(model),
                        epoch,
                        float(means[0]),
                    )
                history.append(
                    dict(
                        stage=stage,
                        epoch=epoch,
                        total_loss=float(means[0]),
                        spectral_mse=float(means[1]),
                        parameter_loss=float(means[2]),
                        elapsed_s=time.perf_counter() - tick,
                    )
                )
                print(
                    "Stage %d epoch %d/%d: total=%.6g spectral=%.6g parameter=%.6g"
                    % (stage, epoch, settings["stage%d_epochs" % stage], *means),
                    flush=True,
                )
        finally:
            for hook in hooks:
                hook.remove()
        model.load_state_dict(best_state)
        selected["stage%d" % stage] = dict(epoch=best_epoch, objective=best_loss)
        checkpoint = dict(
            schema_version=1,
            state_dict=best_state,
            config=settings,
            wavelength_A=torch.tensor(data.wavelength_A),
            parameter_names=list(PARAMETER_NAMES),
            selected_epochs=dict(selected),
            training_sha256=file_sha256(data_path),
        )
        torch.save(checkpoint, output / ("stage1.pt" if stage == 1 else "model.pt"))
        with (output / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    summary = dict(
        config=settings,
        selected=selected,
        n_spectra=len(indices),
        elapsed_s=time.perf_counter() - started,
        device=str(device),
        vp_metadata=vp_metadata,
        training_sha256=file_sha256(data_path),
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary
