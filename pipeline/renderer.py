from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QImage, QPixmap, QFont
from PyQt5.QtWidgets import (QWidget, QToolTip, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem)

from pipeline.metrics import execute_hits_hough
from pipeline.pipeline_processor import SETIPipelineProcessor
from utils.det_utils import decode_F, nms_1d, plot_F_lines


class SETIWaterfallRenderer(QWidget):
    def __init__(self, dataset, model, device, mode='mask', log_dir=Path("./pipeline/log"),
                 verbose=False, parent=None, drift=[0.05, 4.0], snr_threshold=5.0,
                 min_abs_drift=0.05, nms_iou_thresh=0.5, nms_score_thresh=0.5, nms_top_k=10):
        """
        Initialize the SETI waterfall renderer with dataset and model

        Args:
            dataset: SETIDataset instance
            model: Trained DWTNet model
            device: Computation device (e.g., 'cuda' or 'cpu')
            mode: 'mask' or 'detection' - operating mode
            log_dir: Directory for logs
            verbose: Whether to use simple console output
            drift: Drift rate range for mask mode [min_drift, max_drift] in Hz/s
            snr_threshold: SNR threshold for mask mode hits detection
            min_abs_drift: Minimum absolute drift rate for mask mode
            nms_iou_thresh: IoU threshold for detection mode NMS
            nms_score_thresh: Confidence threshold for detection mode NMS
            nms_top_k: Maximum number of detections to keep per patch
        """
        super().__init__(parent)
        self.mode = mode
        self.drift = drift
        self.snr_threshold = snr_threshold
        self.min_abs_drift = min_abs_drift
        self.nms_iou_thresh = nms_iou_thresh
        self.nms_score_thresh = nms_score_thresh
        self.nms_top_k = nms_top_k

        # Initialize processor with mode-specific parameters
        self.processor = SETIPipelineProcessor(
            dataset, model, device, mode=mode, log_dir=log_dir, verbose=verbose,
            drift=drift, snr_threshold=snr_threshold, min_abs_drift=min_abs_drift,
            nms_iou_thresh=nms_iou_thresh, nms_score_thresh=nms_score_thresh, nms_top_k=nms_top_k
        )

        self.dataset = self.processor.dataset
        self.model = self.processor.model
        self.device = self.processor.device

        # Get grid dimensions from processor
        self.grid_height = self.processor.grid_height
        self.grid_width = self.processor.grid_width
        self.cell_size = 12  # Size in pixels for each grid cell

        # Extract tsamp and foff from processor
        self.tsamp = self.processor.tsamp
        self.foff = self.processor.foff

        # Use processor's storage
        self.cell_status = self.processor.cell_status
        self.confidence_scores = self.processor.confidence_scores
        self.freq_ranges = self.processor.freq_ranges
        self.time_ranges = self.processor.time_ranges

        # Current processing position
        self.current_col = 0
        self.current_row = 0

        # Image display related
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(280, 360)  # Increased height for hits info
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.hide()
        self.last_click_pos = None

        # Layout parameters
        self.min_display_height = 400  # Minimum display height in pixels
        self.max_display_height = 1200  # Maximum display height before folding or scrolling
        self.max_display_width = 900  # Maximum display width before wrapping
        self.col_spacing = 20  # Spacing between folded subgrids in a visual row
        self.row_spacing = 10  # Spacing between visual rows of subgrids

        # Folding type
        self.folding_type = 'vertical'

        # Calculate layout for frequency direction (vertical folding for tall grids)
        original_height = self.grid_height * self.cell_size
        self.rows_per_fold = self.max_display_height // self.cell_size
        if original_height > self.max_display_height:
            self.num_folds = (self.grid_height + self.rows_per_fold - 1) // self.rows_per_fold
        else:
            self.num_folds = 1
            self.rows_per_fold = self.grid_height

        self.subgrid_width = self.grid_width * self.cell_size
        self.subgrid_height = self.rows_per_fold * self.cell_size  # Max per fold

        # Calculate wraps
        self.folds_per_line = max(1, (self.max_display_width + self.col_spacing) // (
                self.subgrid_width + self.col_spacing))
        self.num_lines = (self.num_folds + self.folds_per_line - 1) // self.folds_per_line

        # Content size
        content_width = self.folds_per_line * self.subgrid_width + max(0, self.folds_per_line - 1) * self.col_spacing
        content_height = self.num_lines * self.subgrid_height + max(0, self.num_lines - 1) * self.row_spacing

        # Handle min height: if content_height < min, adjust (for now, just set min, but add logic if needed)
        if original_height < self.min_display_height and self.grid_width * self.cell_size > self.max_display_width / 2:
            # Optional: If height too small and width large, fold horizontally to make taller
            # For symmetry, split columns if width large and height small
            self.cols_per_fold = self.max_display_width // self.cell_size
            num_stacks = (self.grid_width + self.cols_per_fold - 1) // self.cols_per_fold
            if num_stacks > 1:
                self.folding_type = 'horizontal'
                self.subgrid_height = self.grid_height * self.cell_size
                self.subgrid_width = self.cols_per_fold * self.cell_size
                self.num_folds = num_stacks
                self.rows_per_fold = self.grid_height  # Full height
                self.folds_per_line = 1  # Stack vertically
                self.num_lines = num_stacks
                content_width = self.subgrid_width
                content_height = num_stacks * self.subgrid_height + (num_stacks - 1) * self.row_spacing
        else:
            # If still < min after calc, could scale or center, but for now proceed
            pass

        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for grid
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(False)
        self.grid_content = GridContent(self, content_width, content_height)
        self.scroll_area.setWidget(self.grid_content)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # Info panel
        self.info_panel = QWidget(self)
        self.info_panel.setFixedWidth(300)
        info_layout = QVBoxLayout(self.info_panel)

        # Title label
        title = QLabel(f"SETI Waterfall Processor ({mode} mode)")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        info_layout.addWidget(title)

        # Processing status label
        self.status_label = QLabel("Processing...")
        info_layout.addWidget(self.status_label)

        # Progress percentage label
        self.progress_label = QLabel("Progress: 0%")
        info_layout.addWidget(self.progress_label)

        # Statistics display
        stats_layout = QHBoxLayout()
        self.processed_label = QLabel("Processed: 0")
        self.detections_label = QLabel("Detections: 0")
        stats_layout.addWidget(self.processed_label)
        stats_layout.addWidget(self.detections_label)
        info_layout.addLayout(stats_layout)

        # Confidence threshold info
        if self.mode == 'mask':
            self.confidence_label = QLabel("Detection Threshold: >0.8 or <0.2")
        else:
            self.confidence_label = QLabel(f"Detection Threshold: >{nms_score_thresh}")
        info_layout.addWidget(self.confidence_label)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_layout.addWidget(self.pause_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_grid)
        btn_layout.addWidget(reset_btn)
        info_layout.addLayout(btn_layout)

        # Image display section
        cell_detail_label = QLabel("Cell Detail:")
        cell_detail_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(cell_detail_label)
        info_layout.addWidget(self.image_label)
        info_layout.addStretch()

        # Hits information display (below the image)
        hits_title = QLabel("Hits Info:")
        hits_title.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(hits_title)
        self.hits_table = QTableWidget(self)
        self.hits_table.setWordWrap(False)
        self.hits_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.hits_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        font = QFont()
        font.setPointSize(9)  # Smaller font size
        self.hits_table.setFont(font)
        self.hits_table.horizontalHeader().setFont(font)
        self.hits_table.verticalHeader().setFont(font)
        info_layout.addWidget(self.hits_table)
        info_layout.addStretch()

        main_layout.addWidget(self.info_panel)

        # Set initial window size to enable scrolling if necessary
        self.resize(content_width + 300, min(content_height, 800))

        # Setup processing timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_cell)
        self.timer.start(50)  # Processing speed in milliseconds

        self.setMouseTracking(True)  # Enable mouse tracking for tooltips

    def toggle_pause(self, paused):
        """
        Toggle processing pause state

        Args:
            paused (bool): True to pause processing, False to resume
        """
        if paused:
            self.timer.stop()
            self.pause_btn.setText("Resume")
            self.status_label.setText("Paused")
        else:
            self.timer.start(50)
            self.pause_btn.setText("Pause")
            self.status_label.setText("Processing...")

    def reset_grid(self):
        """Reset the grid to its initial state"""
        self.current_col = 0
        self.current_row = 0
        self.processor.reset()

        self.cell_status = self.processor.cell_status
        self.confidence_scores = self.processor.confidence_scores
        self.freq_ranges = self.processor.freq_ranges
        self.time_ranges = self.processor.time_ranges

        self.image_label.hide()
        self.grid_content.update()
        self.timer.start(50)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        self.status_label.setText("Processing...")
        self.update_stats()

    def process_next_cell(self):
        """Process the next cell in the grid"""
        if self.current_row < self.grid_height:
            # Process using processor
            self.processor.process_patch(self.current_row, self.current_col)

            # Update progress
            self.update_progress()

            # Move to next cell
            self.current_col += 1
            if self.current_col >= self.grid_width:
                self.current_col = 0
                self.current_row += 1

            self.grid_content.update()  # Trigger repaint
        else:
            self.timer.stop()
            self.status_label.setText("Processing Complete!")

    def update_progress(self):
        """Update progress information in the UI"""
        processed = self.current_row * self.grid_width + self.current_col
        total = self.grid_width * self.grid_height
        progress = int(100 * processed / total)
        self.progress_label.setText(f"Progress: {progress}%")
        self.update_stats()

    def update_stats(self):
        """Update processing statistics in the UI"""
        processed = 0
        detections = 0

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self.confidence_scores[row][col] > 0:
                    processed += 1
                    if self.cell_status[row][col] is True:
                        detections += 1

        self.processed_label.setText(f"Processed: {processed}")
        self.detections_label.setText(f"Detections: {detections}")

    def _process_detection_predictions(self, raw_preds, freq_range):
        """
        Process detection mode predictions to extract frequency intervals

        Args:
            raw_preds: Raw detection predictions from model
            freq_range: Frequency range tuple (min_freq, max_freq) in MHz

        Returns:
            pred_boxes_tuple: Tuple (N, f_starts, f_stops) for plotting
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

        if len(pred_boxes) == 0:
            return (0, [], [])

        # Extract start and stop frequencies (normalized)
        f_starts = pred_boxes[:, 0].cpu().numpy()
        f_stops = pred_boxes[:, 1].cpu().numpy()

        return (len(pred_boxes), f_starts, f_stops)

    def show_cell_detail(self, row, col):
        """
        Display detailed information for a cell, including:
        - Original spectrum
        - Denoised spectrum
        - Additional hits info (text) shown in the sidebar
        """
        patch_data, freq_range, time_range_idx = self.dataset.get_patch(row, col)
        freq_min, freq_max = freq_range  # MHz
        time_start, time_end = (time_range_idx[0] * self.tsamp,
                                time_range_idx[1] * self.tsamp)
        tchans = time_range_idx[1] - time_range_idx[0]

        patch_data = patch_data.to(self.device).unsqueeze(0)  # (1, 1, t, f)

        with torch.no_grad():
            if self.mode == 'mask':
                denoised, mask, logits = self.model(patch_data)
                raw_preds = None
            else:  # detection mode
                denoised, raw_preds = self.model(patch_data)
                logits = torch.tensor([0.0], device=self.device)  # Default logits

            denoised_np = denoised.squeeze().cpu().numpy()
        patch_np = patch_data.squeeze().cpu().numpy()

        # High DPI figure for sharper text
        fig, axs = plt.subplots(2, 1, figsize=(8, 8.5), dpi=200)
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95, hspace=0.3)

        # Common plot params
        label_font = 12
        title_font = 13

        # Original spectrum
        im0 = axs[0].imshow(patch_np, aspect='auto', origin='lower',
                            extent=[freq_min, freq_max, time_start, time_end],
                            cmap='viridis')
        axs[0].set_ylabel("Time (seconds)", fontsize=label_font)
        axs[0].set_xlabel("Frequency (MHz)", fontsize=label_font)
        axs[0].set_title(f"Original Spectrum\nCell ({row}, {col})", fontsize=title_font)
        cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        cbar0.set_label("Intensity", fontsize=label_font)

        # Denoised spectrum
        im1 = axs[1].imshow(denoised_np, aspect='auto', origin='lower',
                            extent=[freq_min, freq_max, time_start, time_end],
                            cmap='viridis')
        axs[1].set_ylabel("Time (seconds)", fontsize=label_font)
        axs[1].set_xlabel("Frequency (MHz)", fontsize=label_font)
        axs[1].set_title(f"Denoised Spectrum\nCell ({row}, {col})", fontsize=title_font)
        cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Intensity", fontsize=label_font)

        # Load hits info and draw detected signals
        df_hits = self.load_additional_info(row, col, denoised_np, raw_preds)

        if self.mode == 'mask' and not df_hits.empty:
            # Mask mode: draw Hough transform lines
            for i, (_, hit) in enumerate(df_hits.iterrows()):
                drift_rate = hit['DriftRate']
                uncorr_freq = hit['Uncorrected_Frequency']
                t_vals = np.arange(0, tchans) * self.tsamp  # [s]
                f_abs_Hz = uncorr_freq + drift_rate * t_vals  # Hz
                f_abs_MHz = freq_min - f_abs_Hz * 1e-6  # MHz
                axs[1].plot(f_abs_MHz, t_vals, 'g--', label='Fit' if i == 0 else None, alpha=0.8)
            if not df_hits.empty:
                axs[1].legend()

        elif self.mode == 'detection' and raw_preds is not None:
            # Detection mode: draw frequency interval boundaries
            pred_boxes_tuple = self._process_detection_predictions(raw_preds, freq_range)
            if pred_boxes_tuple[0] > 0:
                # Create frequency array for plotting
                freqs = np.linspace(freq_min, freq_max, 1000)
                plot_F_lines(axs[1], freqs, pred_boxes_tuple, normalized=True,
                             color='red', linestyle='-', linewidth=1.5)

        # Convert frequencies to absolute MHz for display
        if not df_hits.empty:
            df_hits['Uncorrected_Frequency'] = freq_min - (df_hits['Uncorrected_Frequency'] / 1e6)
            if 'freq_start' in df_hits.columns:
                df_hits['freq_start'] = freq_min - (df_hits['freq_start'] / 1e6)
            if 'freq_end' in df_hits.columns:
                df_hits['freq_end'] = freq_min - (df_hits['freq_end'] / 1e6)

        # Convert high-res matplotlib figure to QPixmap
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = QImage(fig.canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)

        # Downscale to QLabel size → keeps sharpness
        pixmap = QPixmap.fromImage(img).scaled(
            270, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(270, 300)

        # Update hits info in the sidebar (display as text)
        self.update_hits_table(df_hits)

        plt.close(fig)

    def update_hits_table(self, df_hits):
        self.hits_table.clear()
        if df_hits is None or df_hits.empty:
            self.hits_table.setRowCount(0)
            self.hits_table.setColumnCount(0)
        else:
            self.hits_table.setRowCount(len(df_hits))

            # Determine columns based on mode
            if self.mode == 'mask':
                columns = ['DriftRate', 'SNR', 'Uncorrected_Frequency', 'freq_start', 'freq_end']
            else:
                columns = ['DriftRate', 'confidence', 'Uncorrected_Frequency', 'freq_start', 'freq_end']

            # Filter columns to only include those present in the DataFrame
            available_columns = [col for col in columns if col in df_hits.columns]
            self.hits_table.setColumnCount(len(available_columns))
            self.hits_table.setHorizontalHeaderLabels(available_columns)

            for i in range(len(df_hits)):
                for j, col in enumerate(available_columns):
                    value = df_hits.iloc[i][col]
                    if col in ['Uncorrected_Frequency', 'freq_start', 'freq_end']:
                        formatted_value = f"{value:.6f}"  # 6 decimal places for frequencies
                    elif col in ['DriftRate', 'SNR', 'confidence']:
                        formatted_value = f"{value:.4f}"  # 4 decimal places for numerical values
                    else:
                        formatted_value = str(value)
                    self.hits_table.setItem(i, j, QTableWidgetItem(formatted_value))
            self.hits_table.resizeColumnsToContents()

    def load_additional_info(self, row, col, denoised_np, raw_preds=None):
        """
        Load additional information about the cell

        Args:
            row (int): Row index
            col (int): Column index
            denoised_np (np.ndarray): Denoised patch data
            raw_preds: Raw predictions for detection mode

        Returns:
            DataFrame: Hits information
        """
        if self.mode == 'mask':
            # Execute hits detection on denoised data using Hough transform
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
        else:  # detection mode
            # Process detection predictions to generate hits
            freq_range = self.freq_ranges[row][col]
            time_range = self.time_ranges[row][col]
            time_duration = time_range[1] - time_range[0]

            df_hits = self.processor._process_detection_hits(raw_preds, freq_range, time_range)

        return df_hits


class GridContent(QWidget):
    def __init__(self, parent, content_width, content_height):
        super().__init__(parent)
        self.renderer = parent
        self.setFixedSize(content_width, content_height)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        """
        Paint the grid based on cell status and confidence, with folding and wrapping
        """
        painter = QPainter(self)

        UNPROCESSED = QColor(30, 30, 30)
        DETECTED_HIGH_CONF = QColor(0, 200, 0)
        DETECTED_LOW_CONF = QColor(200, 200, 0)
        NO_SIGNAL_HIGH_CONF = QColor(200, 0, 0)
        NO_SIGNAL_LOW_CONF = QColor(200, 200, 0)
        UNCERTAIN = QColor(200, 200, 0)

        if self.renderer.folding_type == 'vertical':
            for fold_i in range(self.renderer.num_folds):
                visual_col = fold_i % self.renderer.folds_per_line
                visual_row = fold_i // self.renderer.folds_per_line

                subgrid_x = visual_col * (self.renderer.subgrid_width + self.renderer.col_spacing)
                subgrid_y = visual_row * (self.renderer.subgrid_height + self.renderer.row_spacing)

                start_row = fold_i * self.renderer.rows_per_fold
                end_row = min(start_row + self.renderer.rows_per_fold, self.renderer.grid_height)

                for r in range(start_row, end_row):
                    visual_r = r - start_row
                    for c in range(self.renderer.grid_width):
                        rect_x = subgrid_x + c * self.renderer.cell_size
                        rect_y = subgrid_y + visual_r * self.renderer.cell_size

                        status = self.renderer.cell_status[r][c]
                        confidence = self.renderer.confidence_scores[r][c]

                        if status is None:
                            if confidence > 0:
                                color = UNCERTAIN
                            else:
                                color = UNPROCESSED
                        elif status:
                            if self.renderer.mode == 'mask':
                                color = DETECTED_HIGH_CONF if confidence > 0.8 else UNCERTAIN
                            else:  # detection mode
                                color = DETECTED_HIGH_CONF if confidence > self.renderer.nms_score_thresh else UNCERTAIN
                        else:
                            if self.renderer.mode == 'mask':
                                color = NO_SIGNAL_HIGH_CONF if confidence < 0.8 else UNCERTAIN
                            else:  # detection mode
                                color = NO_SIGNAL_HIGH_CONF if confidence <= self.renderer.nms_score_thresh else UNCERTAIN

                        # Fill cell with appropriate color
                        painter.fillRect(rect_x, rect_y, self.renderer.cell_size, self.renderer.cell_size, color)

                        # Draw cell border
                        painter.setPen(QColor(60, 60, 60))
                        painter.drawRect(rect_x, rect_y, self.renderer.cell_size, self.renderer.cell_size)

        elif self.renderer.folding_type == 'horizontal':
            for fold_i in range(self.renderer.num_folds):
                visual_col = fold_i % self.renderer.folds_per_line
                visual_row = fold_i // self.renderer.folds_per_line

                subgrid_x = visual_col * (self.renderer.subgrid_width + self.renderer.col_spacing)
                subgrid_y = visual_row * (self.renderer.subgrid_height + self.renderer.row_spacing)

                start_col = fold_i * self.renderer.cols_per_fold
                end_col = min(start_col + self.renderer.cols_per_fold, self.renderer.grid_width)

                for r in range(self.renderer.grid_height):
                    for c in range(start_col, end_col):
                        visual_c = c - start_col
                        rect_x = subgrid_x + visual_c * self.renderer.cell_size
                        rect_y = subgrid_y + r * self.renderer.cell_size

                        status = self.renderer.cell_status[r][c]
                        confidence = self.renderer.confidence_scores[r][c]

                        if status is None:
                            if confidence > 0:
                                color = UNCERTAIN
                            else:
                                color = UNPROCESSED
                        elif status:
                            if self.renderer.mode == 'mask':
                                color = DETECTED_HIGH_CONF if confidence > 0.8 else UNCERTAIN
                            else:  # detection mode
                                color = DETECTED_HIGH_CONF if confidence > self.renderer.nms_score_thresh else UNCERTAIN
                        else:
                            if self.renderer.mode == 'mask':
                                color = NO_SIGNAL_HIGH_CONF if confidence < 0.2 else UNCERTAIN
                            else:  # detection mode
                                color = NO_SIGNAL_HIGH_CONF if confidence <= self.renderer.nms_score_thresh else UNCERTAIN

                        painter.fillRect(rect_x, rect_y, self.renderer.cell_size, self.renderer.cell_size,
                                         color)

                        # Draw cell border
                        painter.setPen(QColor(60, 60, 60))
                        painter.drawRect(rect_x, rect_y, self.renderer.cell_size, self.renderer.cell_size)

    def mouseMoveEvent(self, event):
        """
        Handle mouse movement to show tooltips with frequency information
        """
        pos = event.pos()
        fold_i, r, c = self.map_to_grid(pos.x(), pos.y())
        if fold_i is not None:
            status = self.renderer.cell_status[r][c]
            confidence = self.renderer.confidence_scores[r][c]
            freq_min, freq_max = self.renderer.freq_ranges[r][c]
            time_start, time_end = self.renderer.time_ranges[r][c]

            if status is None:
                status_str = "Not processed"
                if confidence > 0:  # If processed but uncertain
                    status_str = f"Uncertain (Confidence: {confidence:.2f})"
            elif status is True:
                status_str = f"Signal detected (Confidence: {confidence:.2f})"
            else:
                status_str = f"No signal (Confidence: {confidence:.2f})"

            tooltip = (f"Cell ({r}, {c})\n"
                       f"Status: {status_str}\n"
                       f"Frequency: {freq_min:.4f} - {freq_max:.4f} MHz\n"
                       f"Time: {time_start:.2f} - {time_end:.2f} seconds\n"
                       f"Mode: {self.renderer.mode}")

            QToolTip.showText(event.globalPos(), tooltip)

    def mousePressEvent(self, event):
        """
        Handle mouse clicks to show cell details and denoised spectrum
        """
        pos = event.pos()
        fold_i, r, c = self.map_to_grid(pos.x(), pos.y())
        if fold_i is not None and self.renderer.confidence_scores[r][c] > 0:
            self.renderer.last_click_pos = (r, c)
            self.renderer.show_cell_detail(r, c)
            self.renderer.image_label.show()
        else:
            self.renderer.image_label.hide()

    def map_to_grid(self, x, y):
        """
        Map pixel position to grid row and column, considering folds and wraps

        Returns:
            fold_i, row, col or None, None, None if out of bounds
        """
        if self.renderer.folding_type == 'vertical':
            for fold_i in range(self.renderer.num_folds):
                visual_col = fold_i % self.renderer.folds_per_line
                visual_row = fold_i // self.renderer.folds_per_line

                subgrid_x = visual_col * (self.renderer.subgrid_width + self.renderer.col_spacing)
                subgrid_y = visual_row * (self.renderer.subgrid_height + self.renderer.row_spacing)

                if subgrid_x <= x < subgrid_x + self.renderer.subgrid_width and \
                        subgrid_y <= y < subgrid_y + self.renderer.subgrid_height:

                    local_x = x - subgrid_x
                    local_y = y - subgrid_y

                    c = local_x // self.renderer.cell_size
                    visual_r = local_y // self.renderer.cell_size

                    if 0 <= c < self.renderer.grid_width:
                        start_row = fold_i * self.renderer.rows_per_fold
                        r = start_row + visual_r
                        if start_row <= r < min(start_row + self.renderer.rows_per_fold, self.renderer.grid_height):
                            return fold_i, r, c
        elif self.renderer.folding_type == 'horizontal':
            for fold_i in range(self.renderer.num_folds):
                visual_col = fold_i % self.renderer.folds_per_line
                visual_row = fold_i // self.renderer.folds_per_line

                subgrid_x = visual_col * (self.renderer.subgrid_width + self.renderer.col_spacing)
                subgrid_y = visual_row * (self.renderer.subgrid_height + self.renderer.row_spacing)

                if subgrid_x <= x < subgrid_x + self.renderer.subgrid_width and \
                        subgrid_y <= y < subgrid_y + self.renderer.subgrid_height:

                    local_x = x - subgrid_x
                    local_y = y - subgrid_y

                    visual_c = local_x // self.renderer.cell_size
                    r = local_y // self.renderer.cell_size

                    if 0 <= r < self.renderer.grid_height:
                        start_col = fold_i * self.renderer.cols_per_fold
                        c = start_col + visual_c
                        if start_col <= c < min(start_col + self.renderer.cols_per_fold, self.renderer.grid_width):
                            return fold_i, r, c

        return None, None, None
