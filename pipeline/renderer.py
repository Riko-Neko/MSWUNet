from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QImage, QPixmap
from PyQt5.QtWidgets import (QWidget, QToolTip, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea)

from pipeline.metrics import execute_hits
from pipeline.pipeline_processor import SETIPipelineProcessor


class SETIWaterfallRenderer(QWidget):
    def __init__(self, dataset, model, device, log_dir=Path("./pipeline/log"), simple_output=False, parent=None):
        """
        Initialize the SETI waterfall renderer with dataset and model

        Args:
            dataset: SETIDataset instance
            model: Trained DWTNet model
            device: Computation device (e.g., 'cuda' or 'cpu')
            log_dir: Directory for logs
            simple_output: Whether to use simple console output
        """
        super().__init__(parent)
        self.processor = SETIPipelineProcessor(dataset, model, device, log_dir=log_dir, simple_output=simple_output)
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
        title = QLabel("SETI Waterfall Processor")
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

        # Confidence threshold slider info
        self.confidence_label = QLabel("Confidence Threshold: 0.8")
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
        info_layout.addWidget(QLabel("Cell Detail:"))
        info_layout.addWidget(self.image_label)
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
        self.cell_status = [[None for _ in range(self.grid_width)]
                            for _ in range(self.grid_height)]
        self.confidence_scores = [[0.0 for _ in range(self.grid_width)]
                                  for _ in range(self.grid_height)]
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

        # Count processed cells and detections
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self.cell_status[row][col] is not None:
                    processed += 1
                    if self.cell_status[row][col] is True:
                        detections += 1

        self.processed_label.setText(f"Processed: {processed}")
        self.detections_label.setText(f"Detections: {detections}")

    def show_cell_detail(self, row, col):
        """
        Display detailed information for a cell, including denoised spectrum

        Args:
            row (int): Row index
            col (int): Column index
        """
        # Get data for this cell
        patch_data, freq_range, time_range_idx = self.dataset.get_patch(row, col)
        freq_min, freq_max = freq_range
        time_start, time_end = (time_range_idx[0] * self.tsamp, time_range_idx[1] * self.tsamp)

        # Move data to device
        patch_data = patch_data.to(self.device).unsqueeze(0)  # (1, 1, t, f)

        # Run model inference again to get denoised output
        with torch.no_grad():
            denoised, mask, logits = self.model(patch_data)
            denoised_np = denoised.squeeze().cpu().numpy()

        # Create figure with subplots (denoised on top, hits info below)
        fig, axs = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [4, 1]})
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1)  # Minimize margins

        # Plot denoised spectrum
        im = axs[0].imshow(denoised_np, aspect='auto', origin='lower',
                           extent=[time_start, time_end, freq_min, freq_max],
                           cmap='viridis')
        axs[0].set_xlabel("Time (seconds)")
        axs[0].set_ylabel("Frequency (MHz)")
        axs[0].set_title(f"Denoised Spectrum\nCell ({col}, {row})")

        # Add colorbar
        cbar = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")

        # Load and display additional info (hits)
        hits_text = self.load_additional_info(row, col, denoised_np)
        axs[1].axis('off')
        axs[1].text(0.01, 0.95, hits_text, va='top', ha='left', wrap=True, fontsize=8)

        # Convert matplotlib figure to QPixmap
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = QImage(fig.canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(img)

        # Set pixmap to label
        self.image_label.setPixmap(pixmap)

        # Clean up
        plt.close(fig)

    def load_additional_info(self, row, col, denoised_np):
        """
        Load additional information about the cell using metrics.py

        Args:
            row (int): Row index
            col (int): Column index
            denoised_np (np.ndarray): Denoised patch data

        Returns:
            str: Formatted hits information
        """
        # Execute hits detection on denoised data
        df_hits = execute_hits(
            patch=denoised_np,
            tsamp=self.tsamp,
            foff=self.foff,
            max_drift=4.0,
            min_drift=0.1,
            snr_threshold=10.0
        )

        if df_hits.empty:
            return "No hits detected."
        else:
            return "Detected Hits:\n" + df_hits.to_string(index=False)


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

                        # Determine cell color based on status and confidence
                        status = self.renderer.cell_status[r][c]
                        confidence = self.renderer.confidence_scores[r][c]

                        if status is None:  # Not processed or uncertain
                            color = QColor(30, 30, 30)  # Dark gray
                        elif status is True:  # Signal detected
                            if confidence > 0.8:
                                color = QColor(0, 200, 0)  # Green (high confidence)
                            else:
                                color = QColor(200, 200, 0)  # Yellow (medium confidence)
                        else:  # No signal
                            if confidence < 0.2:
                                color = QColor(200, 0, 0)  # Red (high confidence)
                            else:
                                color = QColor(200, 200, 0)  # Yellow (medium confidence)

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

                        # Determine cell color based on status and confidence
                        status = self.renderer.cell_status[r][c]
                        confidence = self.renderer.confidence_scores[r][c]

                        if status is None:  # Not processed or uncertain
                            color = QColor(30, 30, 30)  # Dark gray
                        elif status is True:  # Signal detected
                            if confidence > 0.8:
                                color = QColor(0, 200, 0)  # Green (high confidence)
                            else:
                                color = QColor(200, 200, 0)  # Yellow (medium confidence)
                        else:  # No signal
                            if confidence < 0.2:
                                color = QColor(200, 0, 0)  # Red (high confidence)
                            else:
                                color = QColor(200, 200, 0)  # Yellow (medium confidence)

                        # Fill cell with appropriate color
                        painter.fillRect(rect_x, rect_y, self.renderer.cell_size, self.renderer.cell_size, color)

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

            tooltip = (f"Cell ({c}, {r})\n"
                       f"Status: {status_str}\n"
                       f"Frequency: {freq_min:.4f} - {freq_max:.4f} MHz\n"
                       f"Time: {time_start:.2f} - {time_end:.2f} seconds")

            QToolTip.showText(event.globalPos(), tooltip)

    def mousePressEvent(self, event):
        """
        Handle mouse clicks to show cell details and denoised spectrum
        """
        pos = event.pos()
        fold_i, r, c = self.map_to_grid(pos.x(), pos.y())
        if fold_i is not None and self.renderer.cell_status[r][c] is not None:
            self.renderer.last_click_pos = (r, c)
            self.renderer.show_cell_detail(r, c)

            # Position image relative to click, but clamped within window
            img_size = self.renderer.image_label.size()
            viewport_pos = self.renderer.scroll_area.mapFromGlobal(event.globalPos())
            img_x = max(0, min(viewport_pos.x() - img_size.width() // 2,
                               self.renderer.scroll_area.width() - img_size.width()))
            img_y = max(0, min(viewport_pos.y() - img_size.height() // 2,
                               self.renderer.scroll_area.height() - img_size.height()))

            self.renderer.image_label.move(img_x, img_y)
            self.renderer.image_label.raise_()
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
