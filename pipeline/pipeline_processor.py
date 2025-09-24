import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from pipeline.metrics import execute_hits_hough
from utils.det_utils import decode_F, nms_1d


class SETIPipelineProcessor:
    def __init__(self, dataset, model, device, mode='mask', log_dir=Path("./pipeline/log"),
                 verbose=False, drift=[-4.0, 4.0], snr_threshold=5.0, min_abs_drift=0.05, nms_iou_thresh=0.5,
                 nms_score_thresh=0.5, nms_top_k=10):
        """
        Initialize the SETI pipeline processor with dataset and model

        Args:
            dataset: SETIWaterFullDataset instance
            model: Trained DWTNet model
            device: Computation device (e.g., 'cuda' or 'cpu')
            mode: 'mask' or 'detection' - operating mode
            log_dir: Directory for log files
            verbose: Whether to use simple console output
            drift: Drift rate range for mask mode [min_drift, max_drift] in Hz/s
            snr_threshold: SNR threshold for mask mode hits detection
            min_abs_drift: Minimum absolute drift rate for mask mode
            nms_iou_thresh: IoU threshold for detection mode NMS
            nms_score_thresh: Confidence threshold for detection mode NMS
            nms_top_k: Maximum number of detections to keep per patch
        """
        assert mode in ['mask', 'detection'], f"Unsupported mode: {mode}"

        self.dataset = dataset
        self.model = model.to(device)
        self.device = device
        self.mode = mode

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
            self.snr_threshold = snr_threshold
            self.min_abs_drift = min_abs_drift
        else:  # detection mode
            self.nms_iou_thresh = nms_iou_thresh
            self.nms_score_thresh = nms_score_thresh
            self.nms_top_k = nms_top_k

        # Logging setup
        if log_dir != Path("./pipeline/log"):
            log_dir = Path("./pipeline/log") / log_dir
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

    def _process_detection_hits(self, raw_preds, freq_range, time_range):
        """
        Process detection mode predictions to generate hits information

        Args:
            raw_preds: Raw detection predictions from model
            freq_range: Frequency range tuple (min_freq, max_freq) in MHz
            time_range: Time range tuple (start_time, end_time) in seconds

        Returns:
            df_hits: DataFrame containing detection hits information
        """
        # Decode predictions
        det_outs = [decode_F(raw) for raw in raw_preds]  # List of (B, N_i, 3)
        det_out = torch.cat(det_outs, dim=1)  # (B, total_N, 3)

        # Apply NMS
        pred_boxes_list = nms_1d(det_out,
                                 iou_thresh=self.nms_iou_thresh,
                                 score_thresh=self.nms_score_thresh,
                                 top_k=self.nms_top_k)

        # Process detections for the first batch item (assuming B=1)
        pred_boxes = pred_boxes_list[0]  # (M, 3) - [f_start, f_stop, confidence]

        hits = []
        freq_min, freq_max = freq_range
        time_start, time_end = time_range
        time_duration = time_end - time_start

        # Convert normalized frequency coordinates to absolute MHz
        for box in pred_boxes:
            f_start_norm, f_stop_norm, confidence = box.tolist()

            # Convert normalized frequencies to absolute frequencies (MHz)
            f_start_abs = freq_min + (freq_max - freq_min) * f_start_norm
            f_stop_abs = freq_min + (freq_max - freq_min) * f_stop_norm

            # Ensure frequency ordering for spatial representation
            if f_start_abs > f_stop_abs:
                f_start_abs, f_stop_abs = f_stop_abs, f_start_abs

            # Calculate drift rate (Hz/s)
            # Drift rate = (frequency change) / time duration
            freq_change_hz = (f_stop_abs - f_start_abs) * 1e6  # Convert MHz to Hz
            drift_rate = freq_change_hz / time_duration if time_duration > 0 else 0.0

            # Determine uncorrected frequency (starting frequency)
            uncorr_freq = f_start_abs if f_start_norm <= f_stop_norm else f_stop_abs

            hits.append({
                'DriftRate': drift_rate,
                'SNR': 0.0,  # SNR not available in detection mode
                'Uncorrected_Frequency': uncorr_freq,
                'freq_start': min(f_start_abs, f_stop_abs),
                'freq_end': max(f_start_abs, f_stop_abs),
                'confidence': confidence
            })

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

        # Prepare data
        patch_data = patch_data.to(self.device).unsqueeze(0)  # (1, 1, t, f)

        # Inference
        with torch.no_grad():
            outputs = self.model(patch_data)

            if self.mode == 'mask':
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

            else:  # detection mode
                # Detection mode: denoised, raw_preds
                if len(outputs) == 2:
                    denoised, raw_preds = outputs
                else:
                    # Handle case where model returns different number of outputs
                    denoised = outputs[0]
                    raw_preds = outputs[1] if len(outputs) > 1 else None

                # Use maximum detection confidence as patch confidence
                confidence = 0.0
                if raw_preds is not None:
                    # Extract confidence scores from raw predictions
                    det_outs = [decode_F(raw) for raw in raw_preds]
                    if det_outs:
                        all_confidences = torch.cat([out[:, :, 2] for out in det_outs], dim=1)
                        confidence = all_confidences.max().item() if all_confidences.numel() > 0 else 0.0

                # Determine status based on confidence
                if confidence > self.nms_score_thresh:
                    status = True
                else:
                    status = False

                # Process detections to generate hits
                hits_info = None
                df_hits = pd.DataFrame()
                if status is True and raw_preds is not None:
                    df_hits = self._process_detection_hits(raw_preds, freq_range, time_range)

            # Prepare hits info for logging
            hits_info = df_hits.to_string(index=False) if not df_hits.empty else "No hits detected."

            # Add hits to file if any detected
            if not df_hits.empty:
                # Add additional columns
                df_hits['cell_row'] = row
                df_hits['cell_col'] = col
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
        log_msg = f"Processed cell ({row}, {col}): {status_str} (Confidence: {confidence:.2f})\n"
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
                if self.mode == 'mask':
                    print(
                        f"[\033[36mHit\033[0m] Found signal {idx + 1}: "
                        f"SNR=\033[32m{hit_row['SNR']:.2f}\033[0m, "
                        f"DriftRate=\033[32m{hit_row['DriftRate']:.4f} Hz/s\033[0m, "
                        f"Freq=\033[32m{hit_row['Uncorrected_Frequency']:.6f}\033[0m Mhz, "
                        f"Range=[{hit_row['freq_start']:.6f}, {hit_row['freq_end']:.6f}] MHz"
                    )
                else:  # detection mode
                    print(
                        f"[\033[36mHit\033[0m] Found signal {idx + 1}: "
                        f"Confidence=\033[32m{hit_row['confidence']:.2f}\033[0m, "
                        f"DriftRate=\033[32m{hit_row['DriftRate']:.4f} Hz/s\033[0m, "
                        f"Freq=\033[32m{hit_row['Uncorrected_Frequency']:.6f}\033[0m Mhz, "
                        f"Range=[{hit_row['freq_start']:.6f}, {hit_row['freq_end']:.6f}] MHz"
                    )

        return {
            'status': status,
            'confidence': confidence,
            'freq_range': freq_range,
            'time_range': time_range,
            'hits_info': hits_info,
            'mode': self.mode
        }

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
