import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen.SETIdataset import DynamicSpectrumDataset
from model.tiny_yolo import TinyYOLO, YOLOv1Loss, decode_predictions  # decode_predictions assumed present
from utils.det_utils import plot_F_lines

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
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ----- collate factory -----
import numpy as _np


def make_collate_fn(use_clean=True):
    def collate_fn_for_yolo(batch):
        images, targets = [], []
        for item in batch:
            # dataset returns (noisy_spec, clean_spec, gt_boxes)
            if len(item) == 3:
                noisy_spec, clean_spec, gt_boxes = item
            else:
                # fallback: take first as image, last as gt
                noisy_spec = item[0]
                clean_spec = item[1] if len(item) > 1 else noisy_spec
                gt_boxes = item[-1]
            img = clean_spec if use_clean else noisy_spec

            if isinstance(img, _np.ndarray):
                img = torch.from_numpy(img).float()
            # ensure channel dimension (C,H,W)
            if img.ndim == 2:
                img = img.unsqueeze(0)
            images.append(img)

            # gt_boxes: Tensor (max_num_signals,5) with NaN rows
            if isinstance(gt_boxes, torch.Tensor):
                valid_mask = ~torch.isnan(gt_boxes).all(dim=-1)
                valid = gt_boxes[valid_mask]
                if valid.numel() == 0:
                    targets.append(torch.empty((0, 5), dtype=torch.float32))
                else:
                    targets.append(valid.float())
            else:
                # try convert
                try:
                    t = torch.tensor(gt_boxes, dtype=torch.float32)
                    valid_mask = ~torch.isnan(t).all(dim=-1)
                    targets.append(t[valid_mask] if valid_mask.any() else torch.empty((0, 5), dtype=torch.float32))
                except Exception:
                    targets.append(torch.empty((0, 5), dtype=torch.float32))

        images = torch.stack(images, dim=0)
        return images, targets

    return collate_fn_for_yolo


# ----- build target map (list of (N,5) -> (B,S,S,5+C)) -----
def build_target_map_from_targets_list(targets_list, S, C, device):
    B = len(targets_list)
    target_map = torch.zeros((B, S, S, 5 + C), dtype=torch.float32, device=device)
    for bi, t in enumerate(targets_list):
        if t is None or t.numel() == 0:
            continue
        t = t.float()
        if t.ndim == 1 and t.shape[0] == 5:
            t = t.unsqueeze(0)
        for box in t:
            cls, cx, cy, w, h = box.tolist()
            cx = max(0.0, min(0.9999, cx))
            cy = max(0.0, min(0.9999, cy))
            cell_x = int(cx * S)
            cell_y = int(cy * S)
            cx_in_cell = cx * S - cell_x
            cy_in_cell = cy * S - cell_y
            target_map[bi, cell_y, cell_x, 0] = 1.0
            target_map[bi, cell_y, cell_x, 1] = cx_in_cell
            target_map[bi, cell_y, cell_x, 2] = cy_in_cell
            target_map[bi, cell_y, cell_x, 3] = w
            target_map[bi, cell_y, cell_x, 4] = h
            target_map[bi, cell_y, cell_x, 5 + int(cls) % C] = 1.0
    return target_map


# ----- train/validate with dynamic steps -----
def train_one_epoch(model, loss_fn, dataloader, optimizer, device, steps_per_epoch=100):
    model.train()
    total_loss = 0.0
    iterator = iter(dataloader)
    for step in tqdm(range(steps_per_epoch), desc="Training"):
        try:
            images, targets_list = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            images, targets_list = next(iterator)

        images = images.to(device)
        preds = model(images)
        target_map = build_target_map_from_targets_list(targets_list, S=model.S, C=model.C, device=device)

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
                images, targets_list = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                images, targets_list = next(iterator)

            images = images.to(device)
            preds = model(images)
            target_map = build_target_map_from_targets_list(targets_list, S=model.S, C=model.C, device=device)
            loss, _ = loss_fn(preds, target_map)
            total_loss += float(loss.item())
    return total_loss / max(1, val_steps)


# ----- inference function aligned to decode_predictions output -----
def inference_and_plot(model, dataloader, device, save_dir, steps=10, conf_thresh=0.25, only_class_to_plot=1):
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(dataloader)

    with torch.no_grad():
        for i in range(steps):
            try:
                images, targets_list = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                images, targets_list = next(iterator)

            imgs = images.to(device)
            preds = model(imgs)  # (B,S,S,out_dim)

            # decode_predictions is expected to return: list of detections per image,
            # each detection as [x1,y1,x2,y2,score,class] with coords normalized (0..1).
            dets_list = decode_predictions(preds, model.S, model.B, conf_thresh=conf_thresh)

            # handle batch (we usually have batch_size=1)
            for bi, dets in enumerate(dets_list):
                # robust parse
                classes = []
                f_starts = []
                f_stops = []
                if dets is None:
                    dets = []
                for d in dets:
                    # d could be list, np.ndarray or tensor
                    arr = np.asarray(d)
                    if arr.size == 0:
                        continue
                    # expect x1,y1,x2,y2,score,class  (class may be absent)
                    if arr.shape[0] >= 6:
                        x1, y1, x2, y2, score, cls = arr[:6]
                    elif arr.shape[0] == 5:
                        x1, y1, x2, y2, score = arr[:5]
                        cls = 1  # fallback
                    else:
                        continue
                    # only plot requested class (e.g., F==1)
                    if only_class_to_plot is not None and int(cls) != int(only_class_to_plot):
                        continue
                    # treat y coords as frequency normalized coords
                    f_starts.append(float(y1))
                    f_stops.append(float(y2))
                    classes.append(int(cls))

                N = len(f_starts)
                # prepare 1D spectrum: mean over time axis
                img_np = imgs[bi].detach().cpu().squeeze().numpy()
                if img_np.ndim == 1:
                    spec_1d = img_np
                elif img_np.ndim == 2:
                    spec_1d = img_np.mean(axis=0)
                else:
                    # if channel>1, average channel dim first then time avg
                    if img_np.ndim == 3:
                        # (C,H,W) -> average channels then average time axis
                        spec_1d = img_np.mean(axis=0).mean(axis=0)
                    else:
                        spec_1d = img_np.reshape(-1)

                freqs = np.arange(len(spec_1d))
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(freqs, spec_1d)
                if N > 0:
                    plot_F_lines(ax, freqs, (N, classes, f_starts, f_stops), normalized=True,
                                 color=['red', 'green'], linestyle='--', linewidth=0.8)
                ax.set_xlabel("Frequency channel")
                ax.set_ylabel("Mean intensity (over time)")
                ax.set_title(f"Inference sample #{i} (Bidx={bi}) N_pred={N}")
                out_path = save_dir / f"infer_{i:03d}_b{bi}.png"
                fig.tight_layout()
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"[\033[32mInfo\033[0m] Saved inference plot to {out_path}")


# ----- main CLI -----
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", "-p", action="store_true", help="Run prediction/inference instead of training")
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--val_steps", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--infer_steps", type=int, default=10)
    parser.add_argument("--use_clean", action="store_true", help="Use clean_spec (default False)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--conf_thresh", type=float, default=0.25)
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

    collate_fn = make_collate_fn(use_clean=args.use_clean)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # model + loss
    model = TinyYOLO(num_classes=1, S=7, B=2).to(device)
    loss_fn = YOLOv1Loss(S=model.S, B=model.B, C=model.C).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                                     min_lr=1e-9)

    if args.predict:
        print("[\033[32mInfo\033[0m] Running predict/inference mode...")
        inference_and_plot(model, val_loader, device, "pred_results/plots/yolo_test", steps=args.infer_steps,
                           conf_thresh=args.conf_thresh, only_class_to_plot=1)
        return

    best_val = float("inf")
    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device, steps_per_epoch=args.steps_per_epoch)
        val_loss = validate(model, loss_fn, val_loader, device, val_steps=args.val_steps)
        scheduler.step(val_loss)

        # save epoch ckpt
        epoch_path = CKPT_DIR / f"epoch_{epoch + 1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
        }, epoch_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = CKPT_DIR / "best.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_path)
            print(f"[\033[32mInfo\033[0m] New best val_loss {val_loss:.6f}, saved to {best_path}")

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val:.6f}")

    print("[\033[32mInfo\033[0m] Training finished. Best val:", best_val)


if __name__ == "__main__":
    main()
