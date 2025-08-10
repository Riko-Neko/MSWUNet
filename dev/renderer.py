import sys
import time
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QApplication, QWidget, QToolTip


class GridRenderer(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_size = 64
        self.cell_size = 10  # 每个格子像素大小
        self.setFixedSize(self.grid_size * self.cell_size,
                          self.grid_size * self.cell_size)

        # 渲染状态矩阵，False 表示未渲染，True 表示已渲染
        self.rendered = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # 用于控制渲染顺序
        self.current_x = 0
        self.current_y = 0

        # 定时器控制每格渲染间隔
        self.timer = QTimer()
        self.timer.timeout.connect(self.render_next_cell)
        self.timer.start(10)  # 每100ms渲染一个格子

        self.setMouseTracking(True)

    def render_next_cell(self):
        if self.current_y < self.grid_size:
            self.rendered[self.current_y][self.current_x] = True
            self.update()  # 调用 paintEvent 重绘

            self.current_x += 1
            if self.current_x >= self.grid_size:
                self.current_x = 0
                self.current_y += 1
        else:
            self.timer.stop()  # 渲染完成，停止定时器

    def paintEvent(self, event):
        painter = QPainter(self)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect_x = x * self.cell_size
                rect_y = y * self.cell_size

                if self.rendered[y][x]:
                    painter.fillRect(rect_x, rect_y, self.cell_size, self.cell_size, QColor(255, 0, 0))
                else:
                    painter.fillRect(rect_x, rect_y, self.cell_size, self.cell_size, QColor(30, 30, 30))

                # 画边框
                painter.setPen(QColor(60, 60, 60))
                painter.drawRect(rect_x, rect_y, self.cell_size, self.cell_size)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        x = pos.x() // self.cell_size
        y = pos.y() // self.cell_size
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            QToolTip.showText(event.globalPos(), f"({x}, {y}) 渲染状态: {self.rendered[y][x]}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GridRenderer()
    win.setWindowTitle("动态渲染格子图")
    win.show()
    sys.exit(app.exec_())