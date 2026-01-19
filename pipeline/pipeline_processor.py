import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from utils.det_utils import decode_F, extract_F_slice
from utils.metrics_utils import execute_hits_hough, SNR_filter


class SETIPipelineProcessor:
    def __init__(self, dataset, model, device, mode='mask', log_dir=Path("./pipeline/log"), verbose=False,
                 raw_output=False, drift=[-4.0, 4.0], snr_threshold=10.0, pad_fraction=0.2, min_abs_drift=0.05,
                 iou_thresh=0.5, score_thresh=0.5, top_k=10, fsnr_threshold=300, top_fraction=0.002, min_pixels=50):
        """
        Initialize the SETI pipeline processor with dataset and model

        Args:
            dataset: SETIWaterFullDataset instance
            model: Trained MSWNet model
            device: Computation device (e.g., 'cuda' or 'cpu')
            mode: 'mask' or 'detection' - operating mode
            log_dir: Directory for log files
            verbose: Whether to use simple console output
            raw_output: Whether to use raw output without post-processing
            drift: Drift rate range for mask mode [min_drift, max_drift] in Hz/s
            snr_threshold: SNR threshold for mask mode hits detection
            pad_fraction: Fraction of extents to pad the events in detection.
            min_abs_drift: Minimum absolute drift rate for mask mode
            iou_thresh: IoU threshold for detection mode NMS
            score_thresh: Confidence threshold for detection mode NMS
            top_k: Maximum number of detections to keep per patch
            fsnr_threshold: SNR threshold for denoised trigger criteria
            top_fraction: The proportion of the brightest pixels participating in the "signal average".
            min_pixels: The minimum number of pixels to be involved in the calculation.

        """
        assert mode in ['mask', 'detection'], f"Unsupported mode: {mode}"

        self.dataset = dataset
        self.ascending = dataset.ascending
        self.model = model.to(device)
        self.device = device
        self.mode = mode
        self.snr_threshold = snr_threshold if not raw_output else 0.0

        # Grid dimensions
        self.grid_height = len(dataset.start_t_list)
        self.grid_width = len(dataset.start_f_list)

        # Extract tsamp and foff
        self.tsamp = dataset.obs.header['tsamp']  # Time resolution in seconds
        freqs = dataset.freqs
        self.foff = (freqs[1] - freqs[0]) * 1e6  # Channel width in Hz

        # Initialize storage
        self.cell_status = [[None for _ in range(self.grid_width)]
                            for _ in range(self.grid_height)]
        self.confidence_scores = [[0.0 for _ in range(self.grid_width)]
                                  for _ in range(self.grid_height)]
        self.freq_ranges = [[(0.0, 0.0) for _ in range(self.grid_width)]
                            for _ in range(self.grid_height)]
        self.time_ranges = [[(0.0, 0.0) for _ in range(self.grid_width)]
                            for _ in range(self.grid_height)]

        # Mode-specific parameters
        if self.mode == 'mask':
            self.drift = drift
            self.min_abs_drift = min_abs_drift
        else:  # detection mode
            self.nms_iou_thresh = iou_thresh if not raw_output else 1.0
            self.nms_score_thresh = score_thresh if not raw_output else 0.0
            self.nms_top_k = top_k if not raw_output else None
            self.fSNR_threshold = fsnr_threshold if not raw_output else 0.0
            self.top_fraction = top_fraction
            self.min_pixels = min_pixels
            self.pad_fraction = pad_fraction

        # Logging setup
        if log_dir != Path("./pipeline/log"):
            log_dir = Path("./pipeline/log") / log_dir
        if raw_output:
            log_dir = Path("./pipeline/log/raw")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Overwrite log file if it exists
        file_handler = logging.FileHandler(log_dir / f"pipeline_{timestamp}.log", mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        handlers = [file_handler]
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            handlers.append(console_handler)

        logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.hits_file = log_dir / f"hits_{timestamp}.dat"
        # Remove existing hits file to override
        if self.hits_file.exists():
            try:
                self.hits_file.unlink()
            except Exception as e:
                self.logger.error(f"Failed to remove existing hits file: {e}")

    def _process_detection_hits(self, raw_preds, freq_range, time_range, events_patch=None):
        """
        Process detection mode predictions to generate hits information

        Args:
            raw_preds: Raw detection predictions from model
            freq_range: Frequency range tuple (min_freq, max_freq) in MHz
            time_range: Time range tuple (start_time, end_time) in seconds
            events_patch: Slice of events from the patch

        Returns:
            df_hits: DataFrame containing detection hits information
        """
        # Decode predictions
        det_outs = decode_F(raw_preds, iou_thresh=self.nms_iou_thresh, score_thresh=self.nms_score_thresh)  # dict
        if det_outs["f_start"].shape[1] == 0:
            return pd.DataFrame()
        f_starts = det_outs["f_start"][0].cpu().numpy()
        f_stops = det_outs["f_end"][0].cpu().numpy()
        confidence = det_outs["confidence"][0].cpu().numpy()
        classes = det_outs["class"][0].cpu().numpy()

        hits = []
        freq_min, freq_max = freq_range
        time_start, time_end = time_range
        time_duration = time_end - time_start

        patch_to_slice = None
        if events_patch is not None:
            try:
                if isinstance(events_patch, torch.Tensor):
                    patch_to_slice = events_patch.detach().float()
                else:
                    patch_to_slice = torch.as_tensor(events_patch, dtype=torch.float32)

                if patch_to_slice.ndim == 3:
                    patch_to_slice = patch_to_slice[0]
                if patch_to_slice.ndim != 2:
                    raise ValueError(
                        f"[\033[31mError\033[0m] input_patch must be 2D after squeeze, got {tuple(patch_to_slice.shape)}")
            except Exception as e:
                self.logger.warning(f"Failed to prepare input patch for SNR estimation: {e}")
                patch_to_slice = None

        # Convert normalized frequency coordinates to absolute MHz
        for class_id, f_start_norm, f_stop_norm, confidence in zip(classes, f_starts, f_stops, confidence):
            if class_id < 0:
                continue
            # Convert normalized frequencies to absolute frequencies (MHz)
            if self.ascending:
                f_start = freq_min + (freq_max - freq_min) * f_start_norm
                f_stop = freq_min + (freq_max - freq_min) * f_stop_norm
            else:
                f_start = freq_max - (freq_max - freq_min) * f_start_norm
                f_stop = freq_max - (freq_max - freq_min) * f_stop_norm

            # Calculate drift rate (Hz/s)
            freq_change_hz = (f_stop - f_start) * 1e6  # Convert MHz to Hz
            drift_rate = freq_change_hz / time_duration if time_duration > 0 else 0.0
            relative_drift_rate = -drift_rate if not self.ascending else drift_rate

            # Determine uncorrected frequency (starting frequency)
            uncorr_freq = f_start

            snr_val = 0.0
            if patch_to_slice is not None:
                try:
                    roi, f_start_pad, f_stop_pad, _, _ = extract_F_slice(patch_to_slice, f_start_norm, f_stop_norm,
                                                                         pad_fraction=self.pad_fraction)
                    snr_val = SNR_filter(roi, mode="dedrift_peak", drift_hz_per_s=relative_drift_rate, df_hz=self.foff,
                                         dt_s=self.tsamp)
                except Exception as e:
                    self.logger.warning(f"Failed to estimate SNR for detection: {e}")
                    snr_val = 0.0
            if self.snr_threshold is not None and self.snr_threshold > 0 and snr_val < self.snr_threshold:
                continue

            hits.append({
                'DriftRate': drift_rate,
                'SNR': snr_val,
                'Uncorrected_Frequency': uncorr_freq,
                'freq_start': min(f_start, f_stop),
                'freq_end': max(f_start, f_stop),
                'class_id': class_id,
                'confidence': confidence
            })

        if not hits:
            return pd.DataFrame()

        return pd.DataFrame(hits)

    def process_patch(self, row, col):
        """
        Process a single patch

        Args:
            row: Row index
            col: Column index

        Returns:
            dict: Processing results including status, confidence, ranges
        """
        # Get data
        patch_data, freq_range, time_range_idx = self.dataset.get_patch(row, col)
        time_range = (time_range_idx[0] * self.tsamp, time_range_idx[1] * self.tsamp)
        raw_patch = patch_data[0]

        # Prepare data
        patch_data = patch_data.to(self.device).unsqueeze(0)  # (1, 1, t, f)

        # Inference
        with torch.no_grad():
            outputs = self.model(patch_data)

            if self.mode == 'detection':
                # Detection mode: denoised, regs_dict
                if len(outputs) == 2:
                    denoised, regs_dict = outputs
                else:
                    # Handle case where model returns different number of outputs
                    denoised = outputs[0]
                    regs_dict = outputs[1] if len(outputs) > 1 else None

                try:
                    patch_snr = SNR_filter(denoised.squeeze(), mode="global_topk", top_fraction=self.top_fraction,
                                           min_pixels=self.min_pixels)
                except Exception:
                    patch_snr = 0.0

                # Use maximum detection confidence as patch confidence
                confidence = 0.0
                if regs_dict is not None:
                    # Extract confidence scores from raw predictions
                    # Decode predictions
                    det_outs = decode_F(regs_dict, iou_thresh=self.nms_iou_thresh,
                                        score_thresh=self.nms_score_thresh)  # dict
                    confidences = det_outs["confidence"][0].cpu()
                    if det_outs:
                        confidence = confidences.max().item() if confidences.numel() > 0 else 0.0

                # Determine status based on confidence
                if (confidence >= self.nms_score_thresh) and (patch_snr >= self.fSNR_threshold):
                    status = True
                else:
                    status = False

                # Process detections to generate hits
                hits_info = None
                df_hits = pd.DataFrame()
                if status is True and regs_dict is not None:
                    df_hits = self._process_detection_hits(regs_dict, freq_range, time_range, events_patch=raw_patch)

            else:  # 'mask' mode as default
                # Mask mode: denoised, mask, logits
                if len(outputs) == 3:
                    denoised, mask, logits = outputs
                else:
                    denoised, mask = outputs
                    logits = torch.tensor([0.0], device=self.device)  # Default logits

                probs = torch.sigmoid(logits)
                confidence = probs.mean().item()

                if confidence > 0.8:
                    status = True
                elif confidence < 0.2:
                    status = False
                else:
                    status = None

                # If signal detected with high confidence, compute hits using Hough transform
                hits_info = None
                df_hits = pd.DataFrame()
                if status is True:
                    denoised_np = denoised.squeeze().cpu().numpy()
                    df_hits = execute_hits_hough(
                        patch=denoised_np,
                        tsamp=self.tsamp,
                        foff=self.foff,
                        max_drift=self.drift[1],
                        min_drift=self.drift[0],
                        snr_threshold=self.snr_threshold,
                        min_abs_drift=self.min_abs_drift,
                        merge_tol=10000
                    )

                    if not df_hits.empty:
                        # Convert relative frequencies (in Hz) to absolute frequencies (in MHz)
                        freq_min = freq_range[0]
                        df_hits['Uncorrected_Frequency'] = freq_min - (df_hits['Uncorrected_Frequency'] / 1e6)
                        if 'freq_start' in df_hits.columns:
                            df_hits['freq_start'] = freq_min - (df_hits['freq_start'] / 1e6)
                        if 'freq_end' in df_hits.columns:
                            df_hits['freq_end'] = freq_min - (df_hits['freq_end'] / 1e6)

            # Prepare hits info for logging
            hits_info = df_hits.to_string(index=False) if not df_hits.empty else "No hits detected."

            # Add hits to file if any detected
            if not df_hits.empty:
                # Add additional columns
                df_hits['cell_row'] = row
                df_hits['cell_col'] = col
                df_hits['gSNR'] = patch_snr
                df_hits['freq_min'] = freq_range[0]
                df_hits['freq_max'] = freq_range[1]
                df_hits['time_start'] = time_range[0]
                df_hits['time_end'] = time_range[1]
                df_hits['mode'] = self.mode

                # Append to .dat file (create with header if not exists)
                header = not self.hits_file.exists()
                df_hits.to_csv(self.hits_file, mode='a', sep='\t', header=header, index=False)

        # Store results
        self.cell_status[row][col] = status
        self.confidence_scores[row][col] = confidence
        self.freq_ranges[row][col] = freq_range
        self.time_ranges[row][col] = time_range

        # Log to file (and console if verbose)
        status_str = "Signal detected" if status is True else "No signal" if status is False else "Uncertain"
        log_msg = f"Processed cell ({row}, {col}): {status_str} (Confidence: {confidence:.2f}, Global SNR: {patch_snr:.2f})\n"
        log_msg += f"Frequency: {freq_range[0]:.4f} - {freq_range[1]:.4f} MHz\n"
        log_msg += f"Time: {time_range[0]:.2f} - {time_range[1]:.2f} seconds\n"
        log_msg += f"Mode: {self.mode}"
        if hits_info:
            log_msg += f"\nHits info:\n{hits_info}"
        self.logger.info(log_msg)

        # Flush file handler
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()

        # Simple console output if enabled (only for high-confidence signals with hits)
        if not self.verbose and status is True and not df_hits.empty:
            print(
                f"[\033[35mML\033[0m] Found candidate in cell ({row}, {col}), frequency: {freq_range[0]:.4f} - {freq_range[1]:.4f} MHz")
            for idx, hit_row in df_hits.iterrows():
                if self.mode == 'detection':
                    print(
                        f"[\033[36mHit\033[0m] Found signal {idx + 1}: "
                        f"Class=\033[32m{hit_row['class_id']}\033[0m, "
                        f"Confidence=\033[32m{hit_row['confidence']:.2f}\033[0m, "
                        f"SNR=\033[32m{hit_row['SNR']:.2f}\033[0m, "
                        f"DriftRate=\033[32m{hit_row['DriftRate']:.4f} Hz/s\033[0m, "
                        f"Freq=\033[32m{hit_row['Uncorrected_Frequency']:.6f}\033[0m Mhz, "
                        f"Range=[{hit_row['freq_start']:.6f}, {hit_row['freq_end']:.6f}] MHz"
                    )
                else:  # 'mask' mode as default
                    print(
                        f"[\033[36mHit\033[0m] Found signal {idx + 1}: "
                        f"SNR=\033[32m{hit_row['SNR']:.2f}\033[0m, "
                        f"DriftRate=\033[32m{hit_row['DriftRate']:.4f} Hz/s\033[0m, "
                        f"Freq=\033[32m{hit_row['Uncorrected_Frequency']:.6f}\033[0m Mhz, "
                        f"Range=[{hit_row['freq_start']:.6f}, {hit_row['freq_end']:.6f}] MHz"
                    )

        return status

    def process_all_patches(self):
        """Process all patches in the grid"""
        # Remove existing hits file at start (to override from previous runs)
        if self.hits_file.exists():
            try:
                self.hits_file.unlink()
            except Exception as e:
                self.logger.error(f"Failed to remove existing hits file: {e}")

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                self.process_patch(row, col)
        self.logger.info("All patches processed.")
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()

        # Generate hits CSV sorted by appropriate metric
        csv_file = self.hits_file.with_suffix('.csv')
        if self.hits_file.exists():
            df_all_hits = pd.read_csv(self.hits_file, sep='\t')
            if self.mode == 'mask' and 'SNR' in df_all_hits.columns:
                df_sorted = df_all_hits.sort_values(by='SNR', ascending=False)
            elif self.mode == 'detection' and 'confidence' in df_all_hits.columns:
                df_sorted = df_all_hits.sort_values(by='confidence', ascending=False)
            else:
                df_sorted = df_all_hits
            df_sorted.to_csv(csv_file, index=False)

    def reset(self):
        """Reset the stored status, confidence, and ranges"""
        self.cell_status = [[None for _ in range(self.grid_width)]
                            for _ in range(self.grid_height)]
        self.confidence_scores = [[0.0 for _ in range(self.grid_width)]
                                  for _ in range(self.grid_height)]
        self.freq_ranges = [[(0.0, 0.0) for _ in range(self.grid_width)]
                            for _ in range(self.grid_height)]
        self.time_ranges = [[(0.0, 0.0) for _ in range(self.grid_width)]
                            for _ in range(self.grid_height)]
