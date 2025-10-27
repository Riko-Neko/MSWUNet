import os
import sys
from pathlib import Path

from data_process.post_process.ML import load_ML_dat

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from external.Waterfall import Waterfall

import matplotlib.pyplot as plt
import torch
from utils.det_utils import decode_F, nms_1d, plot_F_lines


def model_crossover_val(wf, dat_df, hit_rows, model, device='cpu', fchans=1024, foff=None, save_dir="visual",
                        mode='mask', nms_iou_thresh=0.5, nms_score_thresh=0.5, nms_top_k=10):
    """
    For a list of TurboSETI hits, grab the corresponding waterfall data,
    run through model, and plot original + denoised spectrum.

    Args:
        wf: Waterfall object
        dat_df: SETI dat DataFrame
        hit_rows: list of row indices in dat_df or dat_df rows themselves
        model: DNN model
        device: 'cpu' or 'cuda'
        fchans: number of frequency channels around the hit to extract
        foff: frequency resolution (MHz per channel)
        save_dir: directory to save plots
        mode: 'mask' or 'detection' - operating mode
        nms_iou_thresh: IoU threshold for detection mode NMS
        nms_score_thresh: Confidence threshold for detection mode NMS
        nms_top_k: Maximum number of detections to keep
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.eval()

    for r in hit_rows:
        if isinstance(r, int):
            row = dat_df.iloc[r]
        else:
            row = r

        # 使用 float64 进行高精度计算
        f_center = np.float64(0.5) * (np.float64(row['freq_start']) + np.float64(row['freq_end']))
        half_bw = (np.float64(fchans) / np.float64(2)) * np.float64(abs(foff))
        f_start = f_center - half_bw
        f_stop = f_center + half_bw
        print(f"[\033[32mInfo\033[0m] Extracting data for row {row.name} from {f_start:.6f} MHz to {f_stop:.6f} MHz")
        # print(row)

        # grab waterfall data
        try:
            freqs, patch_data = wf.grab_data(f_start=f_start, f_stop=f_stop, device=device)
            print(f"[\033[32mInfo\033[0m] Grabbed data for row {row.name}: {patch_data.shape}")
            if patch_data.shape[1] != fchans:
                patch_data = patch_data[:, :fchans]
                freqs = freqs[:fchans]
        except Exception as e:
            print(f"Failed to grab data for row {row.name}: {e}")
            continue

        # 防止 shape 异常
        if patch_data.ndim != 2:
            print(f"Skipping row {row.name}, invalid shape {patch_data.shape}")
            continue

        # 转为 tensor
        mean = np.mean(patch_data)
        std = np.std(patch_data)
        if std < 1e-10:
            std = 1.0
        patch_data = (patch_data - mean) / std
        patch_tensor = torch.from_numpy(patch_data.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            if mode == 'mask':
                denoised, mask, logits = model(patch_tensor)
                raw_preds = None
            else:  # detection mode
                denoised, raw_preds = model(patch_tensor)

            denoised_np = denoised.squeeze().cpu().numpy()
            patch_np = patch_tensor.squeeze().cpu().numpy()

        # 绘制
        fig, axs = plt.subplots(2, 1, figsize=(8, 8.5), dpi=200)
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95, hspace=0.3)
        label_font = 12
        title_font = 13

        time_start, time_end = 0, patch_np.shape[0]
        freq_min, freq_max = freqs[0], freqs[-1]

        # 原始
        im0 = axs[0].imshow(patch_np, aspect='auto', origin='lower',
                            extent=[freq_min, freq_max, time_start, time_end],
                            cmap='viridis')
        axs[0].set_ylabel("Time bins", fontsize=label_font)
        axs[0].set_xlabel("Frequency (MHz)", fontsize=label_font)
        axs[0].set_title(f"Original Spectrum\nRow {row.name}", fontsize=title_font)
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label="Intensity")

        # 去噪
        im1 = axs[1].imshow(denoised_np, aspect='auto', origin='lower',
                            extent=[freq_min, freq_max, time_start, time_end],
                            cmap='viridis')
        axs[1].set_ylabel("Time bins", fontsize=label_font)
        axs[1].set_xlabel("Frequency (MHz)", fontsize=label_font)

        if mode == 'detection':
            axs[1].set_title(f"Denoised Spectrum (Detection Mode)\nRow {row.name}", fontsize=title_font)
        else:
            axs[1].set_title(f"Denoised Spectrum\nRow {row.name}", fontsize=title_font)

        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="Intensity")

        # 在detection模式下绘制检测框
        if mode == 'detection' and raw_preds is not None:
            # 处理检测预测
            det_outs = [decode_F(raw) for raw in raw_preds]  # List of (B, N_i, 3)
            print(f"[\033[32mInfo\033[0m] Detected {len(det_outs[0][0])} candidates for row {row.name}")
            det_out = torch.cat(det_outs, dim=1)  # (B, total_N, 3)

            # 应用NMS
            pred_boxes_list = nms_1d(det_out,
                                     iou_thresh=nms_iou_thresh,
                                     score_thresh=nms_score_thresh,
                                     top_k=nms_top_k)

            # 处理第一个batch的检测结果
            pred_boxes = pred_boxes_list[0]  # (M, 3) - [f_start, f_stop, confidence]

            if len(pred_boxes) > 0:
                # 提取起始和结束频率（归一化）
                f_starts = pred_boxes[:, 0].cpu().numpy()
                f_stops = pred_boxes[:, 1].cpu().numpy()
                confidences = pred_boxes[:, 2].cpu().numpy()

                # 创建检测框元组用于绘图
                pred_boxes_tuple = (len(f_starts), f_starts, f_stops)

                # 使用plot_F_lines绘制检测框
                plot_F_lines(axs[1], freqs, pred_boxes_tuple, normalized=True,
                             color='red', linestyle='-', linewidth=1.5)

                # 在标题中添加检测数量信息
                axs[1].set_title(f"Denoised Spectrum (Detection Mode)\nRow {row.name} - {len(pred_boxes)} detections",
                                 fontsize=title_font)

        # 保存
        fname = save_path / f"row{row.name}_freq{f_center:.6f}_{mode}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved visualization for row {row.name} -> {fname}")


if __name__ == '__main__':
    def load_model(model_class, checkpoint_path, device='cpu', **kwargs):
        """
        Load a PyTorch model from checkpoint.

        Args:
            model_class: The class of the model to instantiate.
            checkpoint_path: Path to the .pth checkpoint file.
            device: 'cpu' or 'cuda'.
            kwargs: Additional kwargs to pass to the model constructor.

        Returns:
            model: Loaded PyTorch model in eval mode on the specified device.
        """
        model = model_class(**kwargs).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        return model


    if __name__ == "__main__":
        # Set random seeds
        torch.manual_seed(42)
        import numpy as np

        np.random.seed(42)

        # Determine device
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"\n[\033[32mInfo\033[0m] Using device: {device}")

        # Checkpoint paths
        # dwtnet_ckpt = Path("../../checkpoints/dwtnet") / "best_model.pth"
        dwtnet_ckpt = Path("../../archived/weights/20250925_33e_det_realbk.pth")

        from model.DWTNet import DWTNet

        dwtnet_args = dict(
            in_chans=1,
            dim=64,
            levels=[2, 4, 8, 16],
            wavelet_name='db4',
            extension_mode='periodization',
            P=2,
            use_spp=True,
            use_pan=True)

        model = load_model(DWTNet, dwtnet_ckpt, device=device, **dwtnet_args)

        mlf = '../../pipeline/log/Kepler-438_M01_pol2_f1120.00-1150.00/hits_20250925_003211.dat'
        # mlf = '../../pipeline/log/HD-180617_M04_pol1_f1400.00-1410.00/hits_20250925_001650.dat'
        setif = '../test_out/truboseti_blis692ns/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.dat'
        # fname = "../../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0.fil"
        # fname = "../../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.fil"
        fname = '../../data/33exoplanets/Kepler-438_M01_pol2_f1120.00-1150.00.fil'
        # fname = '../../data/33exoplanets/HD-180617_M04_pol1_f1400.00-1410.00.fil'

        wf = Waterfall(fname, load_data=True)

        # dat_df = load_seti_dat(setif)
        dat_df = load_ML_dat(mlf, mode='detection')

        # hit_rows = [729, 768, 808, 839, 861]

        # For ML pipeline rows: real row -2
        # hit_rows = [24]
        # hit_rows = [16]
        # hit_rows = list(range(25))  # -1 for list operation
        # hit_rows = list(range(60))
        hit_rows = [10, 18, 25, 29, 32, 33, 34, 35, 36, 37, 43, 44, 45, 46, 47, 51]

        # # 运行mask模式
        # model_crossover_val(wf, dat_df, hit_rows, model, device=device, fchans=1024, foff=wf.header["foff"],
        #                     save_dir=f'visual/{Path(fname).stem}/cross', mode='mask')

        # 运行detection模式
        model_crossover_val(wf, dat_df, hit_rows, model, device=device, fchans=1024, foff=wf.header["foff"],
                            save_dir=f'visual/{Path(fname).stem}/cross', mode='detection', nms_iou_thresh=0.1,
                            nms_score_thresh=0.99, nms_top_k=2)
