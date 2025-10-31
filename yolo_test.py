import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from gen.SETIdataset import DynamicSpectrumDataset
from model.tiny_yolo import TinyYOLO, YOLOv1Loss, decode_F_yolo
from utils.det_utils import plot_F_lines
from utils.loss_func import build_target_yolo

# ======= User configs =======
mode = 'yolo'
tchans = 116
fchans = 1024
df = 7.450580597
dt = 10.200547328
fch1 = None
ascending = True
drift_min = -4.0
drift_max = 4.0
drift_min_abs = df // (tchans * dt)
snr_min = 15.0
snr_max = 25.0
width_min = 10
width_max = 30
num_signals = (0, 2)
noise_std_min = 0.025
noise_std_max = 0.05
noise_mean_min = 2
noise_mean_max = 3
noise_type = "chi2"
use_fil = True
fil_folder = Path('./data/33exoplanets/bk')
background_fil = list(fil_folder.rglob("*.fil"))

# ----- constants & defaults -----
CKPT_DIR = Path("./checkpoints/yolo")
ckpt_path = CKPT_DIR / "best.pth"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ----- train/validate with dynamic steps -----
def train_one_epoch(model, loss_fn, dataloader, optimizer, device, steps_per_epoch=100):
    model.train()
    total_loss = 0.0
    iterator = iter(dataloader)
    for step in tqdm(range(steps_per_epoch), desc="Training"):
        try:
            _, images, targets_list = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            _, images, targets_list = next(iterator)

        images = images.to(device)
        preds = model(images)
        target_map = build_target_yolo(targets_list, S=model.S, C=model.C, device=device)

        optimizer.zero_grad()
        loss, _ = loss_fn(preds, target_map)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
    return total_loss / max(1, steps_per_epoch)


def validate(model, loss_fn, dataloader, device, val_steps=20):
    model.eval()
    total_loss = 0.0
    iterator = iter(dataloader)
    with torch.no_grad():
        for step in tqdm(range(val_steps), desc="Validating"):
            try:
                _, images, targets_list = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                _, images, targets_list = next(iterator)

            images = images.to(device)
            preds = model(images)
            target_map = build_target_yolo(targets_list, S=model.S, C=model.C, device=device)
            loss, _ = loss_fn(preds, target_map)
            total_loss += float(loss.item())
    return total_loss / max(1, val_steps)


# ----- inference function aligned to decode_predictions output -----
# ----- inference function aligned to decode_predictions output -----
def inference_and_plot(model, dataloader, device, save_dir, steps=10, conf_thresh=0.25, iou_threshold=0.5):
    """
    Run inference and plot.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(dataloader)

    with torch.no_grad():
        for i in range(steps):
            try:
                _, images, _ = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                _, images, _ = next(iterator)

            images = images[:1]

            images = images.to(device)
            preds = model(images)

            # decode_predictions returns: list per image of (N, classes, f_starts, f_stops)
            freq_info_list = decode_F_yolo(preds, model.S, model.B, conf_thresh=conf_thresh,
                                           iou_threshold=iou_threshold)


            img_np = images[0].detach().cpu().squeeze().numpy()

            fig, ax = plt.subplots(figsize=(15, 3))
            ax.imshow(img_np, cmap="viridis", aspect="auto", origin="lower")

            N, classes, f_starts, f_stops = freq_info_list[0]
            if N > 0:
                freqs = np.arange(img_np.shape[1])
                plot_F_lines(ax, freqs, (N, classes, f_starts, f_stops), normalized=True, color=['red', 'green'],
                             linestyle='--', linewidth=0.8)

            ax.set_xlabel("Frequency channel")
            ax.set_ylabel("Time channel")
            ax.set_title(f"Inference #{i}")
            out_path = save_dir / f"infer_{i:03d}.png"
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)

            print(f"[\033[32mInfo\033[0m] Saved inference plot to {out_path}")

# ----- main CLI -----
def main(argv=None):
    steps_per_epoch = 500
    val_steps = 200
    epochs = 500
    infer_steps = 10
    batch_size = 1
    conf_thresh = 0.99
    iou_threshold = 0.7
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", "-p", action="store_true", help="Run prediction/inference instead of training")
    args = parser.parse_args() if argv is None else parser.parse_args(argv)

    # device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[\033[32mInfo\033[0m] Using device: {device}")

    # dataset params (reuse your previously set global values)
    train_dataset = DynamicSpectrumDataset(mode='yolo', tchans=tchans, fchans=fchans, df=df, dt=dt,
                                           fch1=None, ascending=ascending, drift_min=drift_min,
                                           drift_max=drift_max, drift_min_abs=drift_min_abs,
                                           snr_min=snr_min, snr_max=snr_max, width_min=width_min,
                                           width_max=width_max, num_signals=num_signals,
                                           noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                           noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                           noise_type=noise_type, use_fil=use_fil, background_fil=background_fil)

    valid_dataset = DynamicSpectrumDataset(mode='yolo', tchans=tchans, fchans=fchans, df=df, dt=dt,
                                           fch1=None, ascending=ascending, drift_min=drift_min,
                                           drift_max=drift_max, drift_min_abs=drift_min_abs,
                                           snr_min=snr_min, snr_max=snr_max, width_min=width_min,
                                           width_max=width_max, num_signals=num_signals,
                                           noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                           noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                           noise_type=noise_type, use_fil=use_fil, background_fil=background_fil)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)

    # model + loss
    model = TinyYOLO(num_classes=2, S=128, B=3).to(device)
    summary(model, input_size=(1, 1, tchans, fchans))

    loss_fn = YOLOv1Loss(S=model.S, B=model.B, C=model.C, lambda_coord=5.0, lambda_noobj=0.05).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    #                                                  min_lr=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1.0e-11)

    if ckpt_path.exists():
        print(f"[\033[32mInfo\033[0m] Loading pretrained weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
    else:
        if args.predict:
            raise ValueError("[\033[31mError\033[0m] No weights found, cannot run inference.")
        print("[\033[32mInfo\033[0m] No pretrained weights found, starting from scratch.")

    if args.predict:
        print("[\033[32mInfo\033[0m] Running predict/inference mode...")

        inference_and_plot(model, val_loader, device, "pred_results/plots/yolo_test", steps=infer_steps,
                           conf_thresh=conf_thresh, iou_threshold=iou_threshold)
        return

    best_val = float("inf")
    for epoch in range(epochs):
        tr_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device, steps_per_epoch=steps_per_epoch)
        val_loss = validate(model, loss_fn, val_loader, device, val_steps=val_steps)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"[\033[32mInfo\033[0m] New best val_loss {val_loss:.6f}, saved to {ckpt_path}")

        print(
            f"Epoch [{epoch + 1}/{epochs}] Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val:.6f}")

    print("[\033[32mInfo\033[0m] Training finished. Best val:", best_val)


if __name__ == "__main__":
    main()
