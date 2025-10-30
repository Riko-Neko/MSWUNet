# train_tinyyolo_yolo_dynamic.py
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen.SETIdataset import DynamicSpectrumDataset
from model.tiny_yolo import TinyYOLO, YOLOv1Loss


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

# ======= Collate function =======
def collate_fn_for_yolo(batch):
    images, targets = [], []
    for item in batch:
        noisy_spec, _, gt_boxes = item  # use clean if you prefer: _, clean, gt_boxes
        images.append(noisy_spec)
        if isinstance(gt_boxes, torch.Tensor):
            valid_mask = ~torch.isnan(gt_boxes).all(dim=-1)
            valid = gt_boxes[valid_mask]
            if valid.numel() == 0:
                targets.append(torch.empty((0, 5), dtype=torch.float32))
            else:
                targets.append(valid.float())
        else:
            targets.append(torch.empty((0, 5), dtype=torch.float32))
    images = torch.stack(images, dim=0)
    return images, targets


# ======= Target map builder =======
def build_target_map_from_targets_list(targets_list, S, C, device):
    B = len(targets_list)
    target_map = torch.zeros((B, S, S, 5 + C), dtype=torch.float32, device=device)
    for bi, t in enumerate(targets_list):
        if t is None or t.numel() == 0:
            continue
        t = t.float()
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


# ======= Train / Validate =======
def train_one_epoch(model, loss_fn, dataloader, optimizer, device, steps_per_epoch=100):
    model.train()
    total_loss = 0
    iterator = iter(dataloader)
    for step in tqdm(range(steps_per_epoch), desc="Training"):
        try:
            _, images, targets_list = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            _, images, targets_list = next(iterator)

        images = images.to(device)
        preds = model(images)
        target_map = build_target_map_from_targets_list(targets_list, S=model.S, C=model.C, device=device)

        optimizer.zero_grad()
        loss, _ = loss_fn(preds, target_map)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
    return total_loss / steps_per_epoch


def validate(model, loss_fn, dataloader, device, val_steps=20):
    model.eval()
    total_loss = 0
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
            target_map = build_target_map_from_targets_list(targets_list, S=model.S, C=model.C, device=device)
            loss, _ = loss_fn(preds, target_map)
            total_loss += float(loss.item())
    return total_loss / val_steps


# ======= Main =======
def main():
    # Set device
    cuda_id = 0
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_id}")
    else:
        device = torch.device("cpu")

    print(f"[\033[32mInfo\033[0m] Using device: {device}")

    train_dataset = DynamicSpectrumDataset(mode=mode, tchans=tchans, fchans=fchans, df=df, dt=dt,
                                           fch1=fch1, ascending=ascending, drift_min=drift_min,
                                           drift_max=drift_max, drift_min_abs=drift_min_abs,
                                           snr_min=snr_min, snr_max=snr_max, width_min=width_min,
                                           width_max=width_max, num_signals=num_signals,
                                           noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                           noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                           noise_type=noise_type, use_fil=use_fil, background_fil=background_fil)

    valid_dataset = DynamicSpectrumDataset(mode=mode, tchans=tchans, fchans=fchans, df=df, dt=dt,
                                           fch1=fch1, ascending=ascending, drift_min=drift_min,
                                           drift_max=drift_max, drift_min_abs=drift_min_abs,
                                           snr_min=snr_min, snr_max=snr_max, width_min=width_min,
                                           width_max=width_max, num_signals=num_signals,
                                           noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                           noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                           noise_type=noise_type, use_fil=use_fil, background_fil=background_fil)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_for_yolo)
    val_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate_fn_for_yolo)

    model = TinyYOLO(num_classes=20, S=7, B=2).to(device)
    loss_fn = YOLOv1Loss(S=model.S, B=model.B, C=model.C).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        tr_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device, steps_per_epoch=100)
        val_loss = validate(model, loss_fn, val_loader, device, val_steps=20)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()