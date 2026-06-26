import sys
import cv2
import os

from PySide6.QtWidgets import (
    QApplication, QLabel, QWidget,
    QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem
)

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt

from detector import run


class MalphasUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MALPHAS // SECURITY DASHBOARD")
        self.resize(1400, 800)

        # =========================================================
        # VIDEO PANEL
        # =========================================================
        self.video_label = QLabel()
        self.video_label.setFixedSize(900, 600)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)

        video_container = QVBoxLayout()

        video_row = QHBoxLayout()
        video_row.addStretch()
        video_row.addWidget(self.video_label)
        video_row.addStretch()

        video_container.addStretch()
        video_container.addLayout(video_row)
        video_container.addStretch()

        # =========================================================
        # LOG PANEL (CARD SYSTEM)
        # =========================================================
        self.log_list = QListWidget()
        self.log_list.setFixedWidth(450)

        self.log_list.setStyleSheet("""
            QListWidget {
                background-color: #0f1116;
                border: none;
            }
        """)

        side_container = QVBoxLayout()
        side_container.addWidget(self.log_list)

        # =========================================================
        # MAIN LAYOUT
        # =========================================================
        hbox = QHBoxLayout()
        hbox.addLayout(video_container)
        hbox.addLayout(side_container)

        self.setLayout(hbox)

        # =========================================================
        # ENGINE
        # =========================================================
        self.engine = run(video_source=0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # =========================================================
    # IMAGE CARD BUILDER
    # =========================================================
    def add_card(self, log):

        item = QListWidgetItem()

        widget = QWidget()
        layout = QHBoxLayout()

        # -------------------------
        # KNOWN FACE IMAGE
        # -------------------------
        known_label = QLabel()

        known_path = log.get("known_face")

        if known_path and os.path.exists(known_path):
            pix = QPixmap(known_path).scaled(
                90,
                90,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            known_label.setPixmap(pix)
            layout.addWidget(known_label)

        # -------------------------
        # DETECTED FACE IMAGE
        # -------------------------
        detected_label = QLabel()

        detected_path = log.get("detected_face")

        if detected_path and os.path.exists(detected_path):
            pix = QPixmap(detected_path).scaled(
                90,
                90,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            detected_label.setPixmap(pix)
            layout.addWidget(detected_label)

        # -------------------------
        # TEXT INFO
        # -------------------------
        info = QLabel(
            f"{log['name']}\n{log['timestamp']}"
        )
        info.setStyleSheet("color: #00ff99; font-size: 11px;")

        # -------------------------
        # CARD LAYOUT
        # -------------------------
        layout.addWidget(known_label)
        layout.addWidget(detected_label)
        layout.addWidget(info)

        widget.setLayout(layout)

        item.setSizeHint(widget.sizeHint())

        self.log_list.addItem(item)
        self.log_list.setItemWidget(item, widget)

    # =========================================================
    # FRAME UPDATE LOOP
    # =========================================================
    def update_frame(self):
        try:
            data = next(self.engine)
        except StopIteration:
            return

        frame = data["frame"]
        fps = data["fps"]
        logs = data["logs"]

        # -------------------------
        # VIDEO UPDATE
        # -------------------------
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.video_label.setPixmap(pix)

        # -------------------------
        # LOG CARDS UPDATE
        # -------------------------
        self.log_list.clear()

        for log in logs:
            # skip FPS entries if any
            if isinstance(log, str):
                continue
            self.add_card(log)


# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MalphasUI()

    app.setStyleSheet("""
        QWidget {
            background-color: #0a0c10;
        }
    """)

    window.show()
    sys.exit(app.exec())