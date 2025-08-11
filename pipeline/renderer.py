from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QImage, QPixmap
from PyQt5.QtWidgets import (QWidget, QToolTip, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton)

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

        # Set window size, leaving space for information panel on the right
        self.setFixedSize(self.grid_width * self.cell_size + 300,
                          self.grid_height * self.cell_size)

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

        # Create information panel on the right
        self.create_info_panel()

        # Setup processing timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_cell)
        self.timer.start(50)  # Processing speed in milliseconds

        self.setMouseTracking(True)  # Enable mouse tracking for tooltips

    def create_info_panel(self):
        """Create the information panel on the right side of the window"""
        right_panel = QWidget(self)
        right_panel.setGeometry(self.grid_width * self.cell_size, 0, 300, self.height())
        layout = QVBoxLayout(right_panel)

        # Title label
        title = QLabel("SETI Waterfall Processor")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        # Processing status label
        self.status_label = QLabel("Processing...")
        layout.addWidget(self.status_label)

        # Progress percentage label
        self.progress_label = QLabel("Progress: 0%")
        layout.addWidget(self.progress_label)

        # Statistics display
        stats_layout = QHBoxLayout()
        self.processed_label = QLabel("Processed: 0")
        self.detections_label = QLabel("Detections: 0")
        stats_layout.addWidget(self.processed_label)
        stats_layout.addWidget(self.detections_label)
        layout.addLayout(stats_layout)

        # Confidence threshold slider info
        self.confidence_label = QLabel("Confidence Threshold: 0.8")
        layout.addWidget(self.confidence_label)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_layout.addWidget(self.pause_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_grid)
        btn_layout.addWidget(reset_btn)
        layout.addLayout(btn_layout)

        # Image display section
        layout.addWidget(QLabel("Cell Detail:"))
        layout.addWidget(self.image_label)
        layout.addStretch()

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
        self.update()
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

            self.update()  # Trigger repaint
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

    def paintEvent(self, event):
        """
        Paint the grid based on cell status and confidence

        Args:
            event: Paint event object
        """
        painter = QPainter(self)

        # Iterate through all cells
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                rect_x = col * self.cell_size
                rect_y = row * self.cell_size

                # Determine cell color based on status and confidence
                status = self.cell_status[row][col]
                confidence = self.confidence_scores[row][col]

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
                painter.fillRect(rect_x, rect_y, self.cell_size, self.cell_size, color)

                # Draw cell border
                painter.setPen(QColor(60, 60, 60))
                painter.drawRect(rect_x, rect_y, self.cell_size, self.cell_size)

    def mouseMoveEvent(self, event):
        """
        Handle mouse movement to show tooltips with frequency information

        Args:
            event: Mouse event object
        """
        pos = event.pos()
        col = pos.x() // self.cell_size
        row = pos.y() // self.cell_size

        # Show tooltip only when inside grid boundaries
        if 0 <= col < self.grid_width and 0 <= row < self.grid_height:
            status = self.cell_status[row][col]
            confidence = self.confidence_scores[row][col]
            freq_min, freq_max = self.freq_ranges[row][col]
            time_start, time_end = self.time_ranges[row][col]

            if status is None:
                status_str = "Not processed"
                if confidence > 0:  # If processed but uncertain
                    status_str = f"Uncertain (Confidence: {confidence:.2f})"
            elif status is True:
                status_str = f"Signal detected (Confidence: {confidence:.2f})"
            else:
                status_str = f"No signal (Confidence: {confidence:.2f})"

            tooltip = (f"Cell ({col}, {row})\n"
                       f"Status: {status_str}\n"
                       f"Frequency: {freq_min:.4f} - {freq_max:.4f} MHz\n"
                       f"Time: {time_start:.2f} - {time_end:.2f} seconds")

            QToolTip.showText(event.globalPos(), tooltip)

    def mousePressEvent(self, event):
        """
        Handle mouse clicks to show cell details and denoised spectrum

        Args:
            event: Mouse event object
        """
        pos = event.pos()
        col = pos.x() // self.cell_size
        row = pos.y() // self.cell_size

        # Ensure click is within grid boundaries
        if not (0 <= col < self.grid_width and 0 <= row < self.grid_height):
            self.image_label.hide()
            return

        # Only show details for processed cells
        if self.cell_status[row][col] is not None:
            self.last_click_pos = (row, col)
            self.show_cell_detail(row, col)

            # Position image centered around click position
            img_size = self.image_label.size()
            img_x = max(0, min(pos.x() - img_size.width() // 2,
                               self.width() - img_size.width()))
            img_y = max(0, min(pos.y() - img_size.height() // 2,
                               self.height() - img_size.height()))

            self.image_label.move(img_x, img_y)
            self.image_label.show()
        else:
            self.image_label.hide()

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
            min_drift=0.0,
            snr_threshold=10.0
        )

        if df_hits.empty:
            return "No hits detected."
        else:
            return "Detected Hits:\n" + df_hits.to_string(index=False)
