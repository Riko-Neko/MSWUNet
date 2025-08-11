import logging
from pathlib import Path

import pandas as pd
import torch

from pipeline.metrics import execute_hits


class SETIPipelineProcessor:
    def __init__(self, dataset, model, device, log_dir=Path("./pipeline/log"), simple_output=False):
        """
        Initialize the SETI pipeline processor with dataset and model

        Args:
            dataset: SETIWaterFullDataset instance
            model: Trained DWTNet model
            device: Computation device (e.g., 'cuda' or 'cpu')
            log_dir: Directory for log files
            simple_output: Whether to use simple console output
        """
        self.dataset = dataset
        self.model = model
        self.device = device

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

        # Logging setup
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "seti_pipeline.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        handlers = [file_handler]
        if not simple_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            handlers.append(console_handler)

        logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
        self.logger = logging.getLogger(__name__)
        self.simple_output = simple_output
        self.hits_file = log_dir / "hits.dat"

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
            denoised, mask, logits = self.model(patch_data)
            probs = torch.sigmoid(logits)
            confidence = probs.mean().item()

            if confidence > 0.8:
                status = True
            elif confidence < 0.2:
                status = False
            else:
                status = None

            # If signal detected with high confidence, compute hits
            hits_info = None
            df_hits = pd.DataFrame()
            if status is True:
                denoised_np = denoised.squeeze().cpu().numpy()
                df_hits = execute_hits(
                    patch=denoised_np,
                    tsamp=self.tsamp,
                    foff=self.foff,
                    max_drift=4.0,
                    min_drift=0.0,
                    snr_threshold=10.0
                )
                hits_info = df_hits.to_string(index=False) if not df_hits.empty else "No hits detected."

                if not df_hits.empty:
                    # Add additional columns
                    df_hits['cell_row'] = row
                    df_hits['cell_col'] = col
                    df_hits['freq_min'] = freq_range[0]
                    df_hits['freq_max'] = freq_range[1]
                    df_hits['time_start'] = time_range[0]
                    df_hits['time_end'] = time_range[1]

                    # Append to .dat file
                    header = not self.hits_file.exists()
                    df_hits.to_csv(self.hits_file, mode='a', sep='\t', header=header, index=False)

        # Store results
        self.cell_status[row][col] = status
        self.confidence_scores[row][col] = confidence
        self.freq_ranges[row][col] = freq_range
        self.time_ranges[row][col] = time_range

        # Log to file (and console if not simple)
        status_str = "Signal detected" if status is True else "No signal" if status is False else "Uncertain"
        log_msg = f"Processed cell ({row}, {col}): {status_str} (Confidence: {confidence:.2f})\n"
        log_msg += f"Frequency: {freq_range[0]:.4f} - {freq_range[1]:.4f} MHz\n"
        log_msg += f"Time: {time_range[0]:.2f} - {time_range[1]:.2f} seconds"
        if hits_info:
            log_msg += f"\nHits info:\n{hits_info}"
        self.logger.info(log_msg)

        # Flush file handler
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()

        # Simple console output if enabled
        if self.simple_output and status is True and not df_hits.empty:
            print(f"Hit found in cell ({row}, {col}) with max SNR: {df_hits['SNR'].max():.2f}")

        return {
            'status': status,
            'confidence': confidence,
            'freq_range': freq_range,
            'time_range': time_range,
            'hits_info': hits_info
        }

    def process_all_patches(self):
        """Process all patches in the grid"""
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                self.process_patch(row, col)
        self.logger.info("All patches processed.")
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()