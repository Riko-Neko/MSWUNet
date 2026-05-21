#!/usr/bin/env python3
"""Export wavelet-internal maps for the manuscript figure.

The script traces the MSWNet reconstruction branch without changing model code.
It uses a hard-coded SETIgen simulation case, loads a checkpoint, and saves
each internal map as a standalone figure.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PROD", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gen.SETIgen import sim_dynamic_spec_seti
from gen.SETIdataset import split_waterfall_generator
from model.DetMSWNet import MSWNet

SIM_PARAMS = {
    "mode": "detection",
    "tchans": 116,
    "fchans": 256,
    "df": 7.450580597,
    "dt": 10.200547328,
    "fch1": None,
    "ascending": True,
    "noise_x_mean": 2.5,
    "noise_x_std": 0.04,
    "noise_type": "chi2",
    "rfi_enhance": True,
}

SIM_SIGNALS = [
    {
        "f_index": 158,
        "drift_rate": 0.05,
        "snr": 12.0,
        "width": 14.0,
        "path": "constant",
        "t_profile": "sine",
        "s_period": 380.0,
        "s_amplitude_factor": 0.35,
        "f_profile": "gaussian",
    }
]

SIM_RFI_PARAMS = {
    "NBC": 10,
    "NBC_amp": 8.0,
    "NBT": 10,
    "NBT_amp": 5.0,
    "BBT": 5,
    "BBT_amp": 10.0,
    "LowDrift": 4,
    "LowDrift_amp_factor": 0.05,
    "LowDrift_width": 5.0,
}

MODEL_ARGS = {
    "in_chans": 1,
    "dim": 64,
    "levels": [2, 4, 8, 16],
    "wavelet_name": "db4",
    "extension_mode": "periodization",
    "fchans": SIM_PARAMS["fchans"],
    "N": 5,
    "num_classes": 2,
    "feat_channels": 64,
    "dropout": 0.005,
}

DEFAULT_CHECKPOINT = "checkpoints/mswunet/bin256/mswnet-bin256-final.pth"
DEFAULT_BACKGROUND_PATTERNS = [
    "data/33exoplanets/yy/*.fil",
    "data/33exoplanets/xx/*.fil",
]
DIVERGING_CMAP = "RdBu_r"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standalone maps for wavelet-constraint visualization."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint path.")
    parser.add_argument(
        "--background",
        action="append",
        default=None,
        help="Real background file, directory, or glob. Can be repeated.",
    )
    parser.add_argument("--out-dir", default="dev/wavelet_constraint_maps", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic sample generation.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument(
        "--reduction",
        choices=("mean", "signed_maxabs", "l2", "channel"),
        default="mean",
        help=(
            "Overview-only reduction for multi-channel maps. The default averages channels; "
            "raw channel exports are controlled separately."
        ),
    )
    parser.add_argument("--channel", type=int, default=0, help="Channel index used when --reduction=channel.")
    parser.add_argument("--dpi", type=int, default=480, help="Figure DPI.")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs="+",
        default=[5.0, 4.0],
        help="Figure size in inches. Use WIDTH HEIGHT, or one value for a square figure.",
    )
    parser.add_argument(
        "--channel-samples",
        "--channel-export-size",
        dest="channel_samples",
        type=int,
        default=16,
        help=(
            "Number of raw feature channels to export for every multi-channel internal map. "
            "Use 0 to disable raw channel export. --channel-export-size is kept as an alias."
        ),
    )
    parser.add_argument(
        "--channel-sample-mode",
        "--channel-export-mode",
        dest="channel_sample_mode",
        choices=("energy", "first", "uniform", "random"),
        default="energy",
        help=(
            "How to select raw channels: highest absolute activation energy, first channels, "
            "uniformly spaced channels, or random channels. --channel-export-mode is kept as an alias."
        ),
    )
    parser.add_argument(
        "--export-lowpass-only",
        action="store_true",
        help=(
            "Run a separate inference pass with all highpass transfers disabled "
            "(high_scales=[0,0,0,0]) and export that full result set into a subdirectory."
        ),
    )
    parser.add_argument(
        "--lowpass-only-subdir",
        default="lowpass_only",
        help="Subdirectory name used under --out-dir for the lowpass-only export set.",
    )
    parser.add_argument("--percentile", type=float, default=99.5, help="Robust percentile for activation scaling.")
    return parser.parse_args()


def parse_figsize(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("--figsize expects one value or two values: WIDTH HEIGHT.")


def apply_prod_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 15,
            "font.weight": "semibold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.unicode_minus": False,
        }
    )


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def collect_background_files(patterns: list[str] | None) -> list[Path]:
    files: list[Path] = []
    for item in patterns or DEFAULT_BACKGROUND_PATTERNS:
        path = resolve_project_path(item)
        if any(ch in str(path) for ch in "*?[]"):
            files.extend(sorted(path.parent.glob(path.name)))
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.fil")))
            files.extend(sorted(path.rglob("*.h5")))
        elif path.is_file():
            files.append(path)

    files = [path for path in files if path.suffix.lower() in {".fil", ".h5"}]
    seen = set()
    unique_files = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(path)

    if not unique_files:
        default_hint = ", ".join(DEFAULT_BACKGROUND_PATTERNS)
        raise FileNotFoundError(
            f"No real background .fil/.h5 files found. Defaults: {default_hint}. "
            "Pass --background with a valid file, directory, or glob."
        )
    return unique_files


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Pass --checkpoint to point to a local final weight file."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)


def normalize_for_model(signal_spec: np.ndarray, clean_spec: np.ndarray, noisy_spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal_mean = float(np.mean(signal_spec))
    signal_std = float(np.std(signal_spec))
    clean_mean = float(np.mean(clean_spec))
    clean_std = float(np.std(clean_spec))
    if signal_std < 1e-10:
        signal_std = 1.0
    if clean_std < 1e-10:
        clean_std = 1.0

    noisy = ((noisy_spec - signal_mean) / signal_std).astype(np.float32)[np.newaxis, :, :]
    clean = ((clean_spec - clean_mean) / clean_std).astype(np.float32)[np.newaxis, :, :]
    return noisy, clean


def boxes_from_freq_info(freq_info: tuple) -> torch.Tensor:
    n_signals, classes, f_starts, f_stops = freq_info
    gt_boxes = torch.full((max(int(n_signals), 1), 3), float("nan"), dtype=torch.float32)
    if n_signals > 0:
        f_den = max(SIM_PARAMS["fchans"] - 1, 1)
        gt_boxes[:n_signals, 0] = torch.tensor(f_starts, dtype=torch.float32) / f_den
        gt_boxes[:n_signals, 1] = torch.tensor(f_stops, dtype=torch.float32) / f_den
        gt_boxes[:n_signals, 2] = torch.tensor(classes, dtype=torch.float32)
    return gt_boxes


def generate_setigen_sample(seed: int, background_files: list[Path]) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    waterfall_itr = split_waterfall_generator(
        background_files,
        SIM_PARAMS["fchans"],
        tchans=SIM_PARAMS["tchans"],
        f_shift=[SIM_PARAMS["fchans"], 8 * SIM_PARAMS["fchans"]],
    )
    signal_spec, clean_spec, noisy_spec, freq_info = sim_dynamic_spec_seti(
        fchans=SIM_PARAMS["fchans"],
        tchans=SIM_PARAMS["tchans"],
        df=SIM_PARAMS["df"],
        dt=SIM_PARAMS["dt"],
        fch1=SIM_PARAMS["fch1"],
        ascending=SIM_PARAMS["ascending"],
        signals=SIM_SIGNALS,
        noise_x_mean=SIM_PARAMS["noise_x_mean"],
        noise_x_std=SIM_PARAMS["noise_x_std"],
        mode=SIM_PARAMS["mode"],
        noise_type=SIM_PARAMS["noise_type"],
        rfi_params=SIM_RFI_PARAMS,
        seed=seed,
        plot=False,
        rfi_enhance=SIM_PARAMS["rfi_enhance"],
        waterfall_itr=waterfall_itr,
    )
    noisy, clean = normalize_for_model(signal_spec, clean_spec, noisy_spec)
    return noisy, clean, boxes_from_freq_info(freq_info)


def scale_high_coeffs(high: list[torch.Tensor] | tuple[torch.Tensor, ...], scale: float) -> list[torch.Tensor]:
    scaled = []
    for item in high:
        scaled.append(None if item is None else item * scale)
    return scaled


def trace_reconstruction_branch(
    model: MSWNet,
    x: torch.Tensor,
    high_scales: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    maps = {
        "input_00_model_input": x,
    }
    if high_scales is None:
        high_scales = [1.0, 1.0, 1.0, 1.0]
    if len(high_scales) != 4:
        raise ValueError(f"Expected 4 highpass scale values, got {len(high_scales)}.")

    lls = []
    highs = []

    e1 = model.enc1(x)
    maps["encoder_01_enc1_before_dwt"] = e1
    ll, high = model.dwt1(e1)
    maps["encoder_01_dwt1_after_lowpass"] = ll
    high = scale_high_coeffs(high, high_scales[0])
    lls.append(ll)
    highs.append(high)

    e2 = model.enc2(ll)
    maps["encoder_02_enc2_before_dwt"] = e2
    ll, high = model.dwt2(e2)
    maps["encoder_02_dwt2_after_lowpass"] = ll
    high = scale_high_coeffs(high, high_scales[1])
    lls.append(ll)
    highs.append(high)

    e3 = model.enc3(ll)
    maps["encoder_03_enc3_before_dwt"] = e3
    ll, high = model.dwt3(e3)
    maps["encoder_03_dwt3_after_lowpass"] = ll
    high = scale_high_coeffs(high, high_scales[2])
    lls.append(ll)
    highs.append(high)

    e4 = model.enc4(ll)
    maps["encoder_04_enc4_before_dwt"] = e4
    ll, high = model.dwt4(e4)
    maps["encoder_04_dwt4_after_lowpass"] = ll
    high = scale_high_coeffs(high, high_scales[3])
    lls.append(ll)
    highs.append(high)

    bottleneck = model.bottleneck(ll)
    maps["bottleneck_00_after_bottleneck"] = bottleneck

    bn = model._interp_if(bottleneck, lls[-1])
    d4_pre_idwt = model.dec4(torch.cat([bn, lls[-1]], dim=1))
    maps["decoder_04_idwt4_input_lowpass"] = d4_pre_idwt
    d4_after_idwt = model.idwt4([d4_pre_idwt, highs[-1]])
    maps["decoder_04_idwt4_output"] = d4_after_idwt
    d4 = model.dec4_c(d4_after_idwt)
    maps["decoder_04_after_dec4_c"] = d4

    d4 = model._interp_if(d4, lls[-2])
    d3_pre_idwt = model.dec3(torch.cat([d4, lls[-2]], dim=1))
    maps["decoder_03_idwt3_input_lowpass"] = d3_pre_idwt
    d3_after_idwt = model.idwt3([d3_pre_idwt, highs[-2]])
    maps["decoder_03_idwt3_output"] = d3_after_idwt
    d3 = model.dec3_c(d3_after_idwt)
    maps["decoder_03_after_dec3_c"] = d3

    d3 = model._interp_if(d3, lls[-3])
    d2_pre_idwt = model.dec2(torch.cat([d3, lls[-3]], dim=1))
    maps["decoder_02_idwt2_input_lowpass"] = d2_pre_idwt
    d2_after_idwt = model.idwt2([d2_pre_idwt, highs[-3]])
    maps["decoder_02_idwt2_output"] = d2_after_idwt
    d2 = model.dec2_c(d2_after_idwt)
    maps["decoder_02_after_dec2_c"] = d2

    d2 = model._interp_if(d2, lls[-4])
    d1_pre_idwt = model.dec1(torch.cat([d2, lls[-4]], dim=1))
    maps["decoder_01_idwt1_input_lowpass"] = d1_pre_idwt

    zero_high = [torch.zeros_like(highs[-4][0])]
    lowpass_only = model.idwt1([d1_pre_idwt, zero_high])
    full_idwt = model.idwt1([d1_pre_idwt, highs[-4]])

    out_features = model.dec1_c(full_idwt)
    cleaned = model.deno_out(out_features)

    maps["decoder_01_idwt1_output"] = full_idwt
    maps["decoder_01_after_dec1_c"] = out_features
    maps["decoder_01_last_lowpass_contribution"] = lowpass_only
    maps["decoder_01_last_full_idwt_synthesis"] = full_idwt
    maps["output_00_cleaned_map"] = cleaned

    return maps


def collapse_to_2d(tensor: torch.Tensor, reduction: str, channel: int) -> np.ndarray:
    x = tensor.detach().float().cpu()
    if x.ndim == 5:
        x = x[0]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        elif reduction == "mean":
            x = x.mean(dim=0)
        elif reduction == "signed_maxabs":
            idx = x.abs().argmax(dim=0, keepdim=True)
            x = x.gather(0, idx).squeeze(0)
        elif reduction == "l2":
            x = torch.sqrt(torch.mean(x * x, dim=0))
        elif reduction == "channel":
            if channel < 0 or channel >= x.shape[0]:
                raise IndexError(f"Channel {channel} out of range for tensor with {x.shape[0]} channels.")
            x = x[channel]
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D map after reduction, got shape {tuple(x.shape)}.")
    return x.numpy()


def tensor_to_channel_stack(tensor: torch.Tensor) -> np.ndarray:
    x = tensor.detach().float().cpu()
    if x.ndim == 5:
        x = x[0]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected a channel stack, got shape {tuple(x.shape)}.")
    return x.numpy()


def select_channel_indices(stack: np.ndarray, count: int, mode: str, seed: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=np.int64)
    count = min(count, stack.shape[0])
    if mode == "first":
        return np.arange(count, dtype=np.int64)
    if mode == "uniform":
        return np.linspace(0, stack.shape[0] - 1, count, dtype=np.int64)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(stack.shape[0], size=count, replace=False)).astype(np.int64)
    scores = np.nanmean(np.abs(stack), axis=(1, 2))
    return np.argsort(scores)[::-1][:count]


def robust_limits(arr: np.ndarray, diverging: bool, percentile: float) -> tuple[float | None, float | None]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    if diverging:
        scale = np.percentile(np.abs(finite), percentile)
        if scale <= 0:
            scale = np.max(np.abs(finite))
        return (-float(scale), float(scale)) if scale > 0 else (None, None)
    lo, hi = np.percentile(finite, [100.0 - percentile, percentile])
    if hi <= lo:
        return None, None
    return float(lo), float(hi)


def save_map(
    arr: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str,
    diverging: bool,
    percentile: float,
    figsize: tuple[float, float],
    dpi: int,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if vmin is None and vmax is None:
        vmin, vmax = robust_limits(arr, diverging=diverging, percentile=percentile)
    render_cmap = DIVERGING_CMAP if diverging else cmap

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap=render_cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Frequency Channel")
    ax.set_ylabel("Time Frame")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.16, right=0.9, bottom=0.14, top=0.86)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_independent_channel_maps(
    stack: np.ndarray,
    channel_indices: np.ndarray,
    out_dir: Path,
    stem: str,
    title: str,
    cmap: str,
    diverging: bool,
    percentile: float,
    figsize: tuple[float, float],
    dpi: int,
) -> int:
    if channel_indices.size == 0:
        return 0
    map_out_dir = out_dir / stem
    map_out_dir.mkdir(parents=True, exist_ok=True)
    selected = stack[channel_indices]
    vmin, vmax = robust_limits(selected, diverging=diverging, percentile=percentile)
    for channel, arr in zip(channel_indices, selected):
        save_map(
            arr,
            map_out_dir / f"ch{int(channel):03d}.png",
            title=f"{title} ch {int(channel)}",
            cmap=cmap,
            diverging=diverging,
            percentile=percentile,
            figsize=figsize,
            dpi=dpi,
            vmin=vmin,
            vmax=vmax,
        )
    return int(channel_indices.size)


def build_image_specs(
    noisy: np.ndarray,
    clean: np.ndarray,
    maps: dict[str, torch.Tensor],
) -> list[tuple[str, torch.Tensor, str, str, bool]]:
    image_specs = [
        ("00_data_noisy_input.png", torch.from_numpy(noisy), "Data: Simulated Noisy Input", "viridis", False),
        ("01_data_clean_target.png", torch.from_numpy(clean), "Data: Synthetic Clean Target", "viridis", False),
        ("02_output_cleaned_map.png", maps["output_00_cleaned_map"], "Output: Cleaned Map", "viridis", False),
    ]

    for level in range(1, 5):
        encoder_lowpass_key = f"encoder_0{level}_dwt{level}_after_lowpass"
        image_specs.extend(
            [
                (
                    f"1{level}_encoder_enc{level}_before_dwt{level}.png",
                    maps[f"encoder_0{level}_enc{level}_before_dwt"],
                    f"Encoder {level}: Before DWT{level}",
                    "coolwarm",
                    True,
                ),
                (
                    f"1{level}_encoder_dwt{level}_after_lowpass.png",
                    maps[encoder_lowpass_key],
                    f"Encoder {level}: DWT{level} Lowpass",
                    "coolwarm",
                    True,
                ),
            ]
        )

    image_specs.append(
        (
            "25_bottleneck_after_bottleneck.png",
            maps["bottleneck_00_after_bottleneck"],
            "Bottleneck: After Bottleneck",
            "coolwarm",
            True,
        )
    )

    for level in range(4, 0, -1):
        decoder_lowpass_key = f"decoder_0{level}_idwt{level}_input_lowpass"
        image_specs.extend(
            [
                (
                    f"3{level}_decoder_idwt{level}_input_lowpass.png",
                    maps[decoder_lowpass_key],
                    f"Decoder {level}: IDWT{level} Lowpass Input",
                    "coolwarm",
                    True,
                ),
                (
                    f"3{level}_decoder_idwt{level}_output.png",
                    maps[f"decoder_0{level}_idwt{level}_output"],
                    f"Decoder {level}: IDWT{level} Output",
                    "coolwarm",
                    True,
                ),
                (
                    f"3{level}_decoder_after_dec{level}_c.png",
                    maps[f"decoder_0{level}_after_dec{level}_c"],
                    f"Decoder {level}: After dec{level}_c",
                    "coolwarm",
                    True,
                ),
            ]
        )

    image_specs.extend(
        [
            (
                "41_final_idwt1_lowpass_contribution.png",
                maps["decoder_01_last_lowpass_contribution"],
                "Final IDWT1: Lowpass Contribution",
                "coolwarm",
                True,
            ),
            (
                "42_final_idwt1_full_synthesis.png",
                maps["decoder_01_last_full_idwt_synthesis"],
                "Final IDWT1: Full Synthesis",
                "coolwarm",
                True,
            ),
        ]
    )
    return image_specs


def export_image_specs(
    image_specs: list[tuple[str, torch.Tensor, str, str, bool]],
    out_dir: Path,
    reduction: str,
    channel: int,
    percentile: float,
    figsize: tuple[float, float],
    dpi: int,
    channel_samples: int,
    channel_sample_mode: str,
    seed: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, tensor, title, cmap, diverging in image_specs:
        arr = collapse_to_2d(tensor, reduction=reduction, channel=channel)
        save_map(
            arr,
            out_dir / filename,
            title=title,
            cmap=cmap,
            diverging=diverging,
            percentile=percentile,
            figsize=figsize,
            dpi=dpi,
        )

    saved_independent_channels = 0
    channel_out_dir = out_dir / "channels"
    for filename, tensor, title, cmap, diverging in image_specs:
        stem = Path(filename).stem
        stack = tensor_to_channel_stack(tensor)
        if stack.shape[0] <= 1:
            continue
        channel_indices = select_channel_indices(
            stack,
            channel_samples,
            channel_sample_mode,
            seed=seed,
        )
        saved_independent_channels += save_independent_channel_maps(
            stack,
            channel_indices,
            channel_out_dir,
            stem,
            title=title,
            cmap=cmap,
            diverging=diverging,
            percentile=percentile,
            figsize=figsize,
            dpi=dpi,
        )
    return saved_independent_channels


def main() -> None:
    args = parse_args()
    figsize = parse_figsize(args.figsize)
    apply_prod_style()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_project_path(args.checkpoint)
    background_files = collect_background_files(args.background)
    noisy, clean, gt_boxes = generate_setigen_sample(args.seed, background_files)

    model = MSWNet(**MODEL_ARGS).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    x = torch.from_numpy(noisy).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        maps = trace_reconstruction_branch(model, x)

    image_specs = build_image_specs(noisy, clean, maps)

    saved_independent_channels = export_image_specs(
        image_specs=image_specs,
        out_dir=out_dir,
        reduction=args.reduction,
        channel=args.channel,
        percentile=args.percentile,
        figsize=figsize,
        dpi=args.dpi,
        channel_samples=args.channel_samples,
        channel_sample_mode=args.channel_sample_mode,
        seed=args.seed,
    )

    lowpass_only_channels = 0
    lowpass_only_dir = None
    if args.export_lowpass_only:
        with torch.no_grad():
            lowpass_only_maps = trace_reconstruction_branch(model, x, high_scales=[0.0, 0.0, 0.0, 0.0])
        lowpass_only_specs = build_image_specs(noisy, clean, lowpass_only_maps)
        lowpass_only_dir = out_dir / args.lowpass_only_subdir
        lowpass_only_channels = export_image_specs(
            image_specs=lowpass_only_specs,
            out_dir=lowpass_only_dir,
            reduction=args.reduction,
            channel=args.channel,
            percentile=args.percentile,
            figsize=figsize,
            dpi=args.dpi,
            channel_samples=args.channel_samples,
            channel_sample_mode=args.channel_sample_mode,
            seed=args.seed,
        )

    n_targets = int(torch.isfinite(gt_boxes[..., 0]).sum().item())
    print(
        f"[Info] Saved {len(image_specs)} maps and {saved_independent_channels} "
        f"raw channel maps to {out_dir}"
    )
    print(f"[Info] SETIgen seed: {args.seed}; finite target rows: {n_targets}")
    print(f"[Info] Real background files: {len(background_files)}")
    print(f"[Info] Checkpoint: {checkpoint_path}")
    if lowpass_only_dir is not None:
        print(
            f"[Info] Saved lowpass-only export to {lowpass_only_dir} "
            f"with {lowpass_only_channels} raw channel maps"
        )


if __name__ == "__main__":
    main()
