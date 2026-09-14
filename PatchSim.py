import stat
import sys
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QFrame,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QMessageBox,
)
import time
from pathlib import Path
import sys
import os
import cv2
import numpy as np
project_root = Path("C:\Soham Projects\Machine Learning - Projects\PCB_Defect_Anaomaly")
from knn import knn_scoring
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from PIL import Image
from Data.transformations import getImageNetTransforms
import torch
from extractor import WideResNetFeatureExtractor   

data_path = project_root / "Data"
embeddings_path = project_root / "memory_bank"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = WideResNetFeatureExtractor().to(device)

DATASET_CATEGORIES = {
    "MVTec AD": {"bottle": 16.14, "cable": 22.35, "capsule": 14.56, "carpet": 18.21, "grid": 23, "hazelnut": 20.39, "leather": 17.85, "metal_nut": 19.97, "pill": 18.71, "screw": 21, "tile": 21, "toothbrush": 18.71, "transistor": 21, "wood": 12, "zipper": 21}
}


class SetupPage(QWidget):
    proceed_clicked = Signal(str, str)

    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(30, 30, 30, 30)
        root.setSpacing(18)

        title = QLabel("PCB anomaly detection")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 22, QFont.Bold))

        subtitle = QLabel("Choose a dataset and category to continue")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 14px;")

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(DATASET_CATEGORIES.keys())
        self.dataset_combo.currentTextChanged.connect(self.update_categories)

        self.category_combo = QComboBox()
        self.update_categories(self.dataset_combo.currentText())

        # self.extractor_combo = QComboBox()
        # self.extractor_combo.addItems(["WideResNet50", "ResNet18", "ViT"])

        self.proceed_btn = QPushButton("Proceed")
        self.proceed_btn.setMinimumHeight(42)
        self.proceed_btn.clicked.connect(self.emit_proceed)

        form = QVBoxLayout()
        form.setSpacing(12)

        form.addWidget(QLabel("Dataset"))
        form.addWidget(self.dataset_combo)

        form.addWidget(QLabel("Category"))
        form.addWidget(self.category_combo)

        # form.addWidget(QLabel("Feature extractor"))
        # form.addWidget(self.extractor_combo)

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addStretch(1)
        root.addLayout(form)
        root.addWidget(self.proceed_btn)
        root.addStretch(2)

        self.setStyleSheet("""
            QComboBox, QPushButton {
                padding: 10px;
                font-size: 14px;
            }
            QPushButton {
                background: #1f6feb;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: #1558b0;
            }
        """)

    def update_categories(self, dataset_name):
        self.category_combo.clear()
        self.category_combo.addItems(DATASET_CATEGORIES.get(dataset_name, []))

    def emit_proceed(self):
        dataset = self.dataset_combo.currentText()
        category = self.category_combo.currentText()

        if not dataset or not category:
            QMessageBox.warning(self, "Missing selection", "Select both dataset and category.")
            return

        self.proceed_clicked.emit(dataset, category)


class UploadPage(QWidget):
    back_clicked = Signal()
    image_selected = Signal(str)

    def __init__(self):
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        top_row = QHBoxLayout()

        self.back_btn = QPushButton("← Back")
        self.back_btn.setMinimumWidth(100)
        self.back_btn.clicked.connect(self.back_clicked.emit)

        top_row.addWidget(self.back_btn)
        top_row.addStretch(1)

        self.context_label = QLabel("Dataset: - | Category: -")
        self.context_label.setStyleSheet("color: #666;")
        top_row.addWidget(self.context_label)

        title = QLabel("Upload Image")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        self.upload_btn = QPushButton("Choose image")
        self.upload_btn.setMinimumHeight(44)
        self.upload_btn.clicked.connect(self.choose_image)

        self.file_label = QLabel("No image selected")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setStyleSheet("color: #666;")

        root.addLayout(top_row)
        root.addWidget(title)
        root.addStretch(1)
        root.addWidget(self.upload_btn)
        root.addWidget(self.file_label)
        root.addStretch(3)

        self.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background: #1f6feb;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: #1558b0;
            }
        """)

    def set_context(self, dataset, category):
        self.context_label.setText(f"Dataset: {dataset} | Category: {category}")

    def choose_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.file_label.setText(file_path)
            self.image_selected.emit(file_path)


class ResultPage(QWidget):
    back_clicked = Signal()

    def __init__(self):
        super().__init__()

        self.current_image_path = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        top_row = QHBoxLayout()
        self.back_btn = QPushButton("← Back")
        self.back_btn.setMinimumWidth(100)
        self.back_btn.clicked.connect(self.back_clicked.emit)

        self.title_label = QLabel("PCB anomaly detection")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)

        top_row.addWidget(self.back_btn)
        top_row.addStretch(1)
        top_row.addWidget(self.title_label)
        top_row.addStretch(2)

        body = QHBoxLayout()
        body.setSpacing(16)

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)

        left_header = QLabel("Input Image")
        left_header.setFont(QFont("Arial", 14, QFont.Bold))
        left_header.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel("Image preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(420, 420)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background: #fafafa;")
        self.image_label.setScaledContents(False)

        left_layout.addWidget(left_header)
        left_layout.addWidget(self.image_label, 1)

        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)

        results_title = QLabel("Results")
        results_title.setFont(QFont("Arial", 14, QFont.Bold))
        results_title.setAlignment(Qt.AlignCenter)

        self.dataset_info = QLabel("Dataset: -")
        self.category_info = QLabel("Category: -")
        self.score_info = QLabel("Anomaly score: -")
        self.prediction_info = QLabel("Prediction: -")
        self.status_info = QLabel("Status: -")

        self.run_again_btn = QPushButton("Upload another image")
        self.run_again_btn.setMinimumHeight(40)

        right_layout.addWidget(results_title)
        right_layout.addSpacing(8)
        right_layout.addWidget(self.dataset_info)
        right_layout.addWidget(self.category_info)
        right_layout.addWidget(self.score_info)
        right_layout.addWidget(self.prediction_info)
        right_layout.addWidget(self.status_info)
        right_layout.addStretch(1)
        right_layout.addWidget(self.run_again_btn)

        body.addWidget(left_panel, 1)
        body.addWidget(right_panel, 1)

        bottom_panel = QFrame()
        bottom_panel.setFrameShape(QFrame.StyledPanel)
        bottom_layout = QVBoxLayout(bottom_panel)

        bottom_title = QLabel("Anomaly Map")
        bottom_title.setFont(QFont("Arial", 14, QFont.Bold))
        bottom_title.setAlignment(Qt.AlignCenter)

        self.map_label = QLabel("Heatmap preview")
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setMinimumHeight(300)
        self.map_label.setStyleSheet("border: 1px solid #ccc; background: #fafafa;")
        self.map_label.setScaledContents(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.hide()

        bottom_layout.addWidget(bottom_title)
        bottom_layout.addWidget(self.map_label, 1)
        bottom_layout.addWidget(self.progress)

        root.addLayout(top_row)
        root.addLayout(body, 3)
        root.addWidget(bottom_panel, 2)

        self.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background: #1f6feb;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: #1558b0;
            }
            QFrame {
                border-radius: 10px;
            }
        """)

    def set_context(self, dataset, category):
        self.dataset_info.setText(f"Dataset: {dataset}")
        self.category_info.setText(f"Category: {category}")

    def set_input_image(self, image_path):
        self.current_image_path = image_path
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def set_heatmap(self, heatmap):
        heatmap = np.asarray(heatmap)

        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        h, w, ch = heatmap_color.shape
        bytes_per_line = ch * w

        qimg = QImage(
            heatmap_color.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qimg)
        self.map_label.setPixmap(
            pixmap.scaled(
                self.map_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def set_results(self, score=None, prediction=None, status=None):
        self.score_info.setText(f"Anomaly score: {score if score is not None else '-'}")
        self.prediction_info.setText(f"Prediction: {prediction if prediction is not None else '-'}")
        self.status_info.setText(f"Status: {status if status is not None else '-'}")

    def resizeEvent(self, event):
        if self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            if not pixmap.isNull():
                self.image_label.setPixmap(
                    pixmap.scaled(
                        self.image_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )
        super().resizeEvent(event)


class PCBApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PCB anomaly detection")
        self.resize(1200, 850)

        self.dataset = None
        self.category = None
        self.image_path = None

        self.extractor = WideResNetFeatureExtractor().to(device)
        self.extractor.eval()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        self.setup_page = SetupPage()
        self.upload_page = UploadPage()
        self.result_page = ResultPage()

        self.stack.addWidget(self.setup_page)
        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.result_page)

        self.setup_page.proceed_clicked.connect(self.go_to_upload)
        self.upload_page.back_clicked.connect(self.go_to_setup)
        self.upload_page.image_selected.connect(self.go_to_results)
        self.result_page.back_clicked.connect(self.go_to_upload)
        self.result_page.run_again_btn.clicked.connect(self.go_to_upload)

    def go_to_setup(self):
        self.stack.setCurrentWidget(self.setup_page)

    def go_to_upload(self):
        self.dataset = self.setup_page.dataset_combo.currentText()
        self.category = self.setup_page.category_combo.currentText()
        self.threshold = DATASET_CATEGORIES[self.dataset][self.category]
        memory_bank = torch.load(embeddings_path / f"memory_bank_{self.category}.pt", map_location="cpu")
        num_samples = int(0.075 * len(memory_bank))
        indices = torch.randperm(len(memory_bank))[:num_samples]
        memory_bank_small = memory_bank[indices]

        pca_dim = 128
        self.pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=42)
        self.memory_bank_pca = self.pca.fit_transform(memory_bank_small.numpy())

        self.nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(self.memory_bank_pca)

        self.upload_page.set_context(self.dataset, self.category)
        self.stack.setCurrentWidget(self.upload_page)

    def go_to_results(self, image_path):
        self.image_path = image_path

        img = Image.open(self.image_path).convert("RGB")
        transform = getImageNetTransforms()
        self.img_tensor = transform(img)
        scores, heatmap = knn_scoring(self.img_tensor, self.memory_bank_pca, self.nn_model, self.extractor, self.pca)
        score = np.partition(scores, -50)[-50:].mean()
        self.result_page.set_context(self.dataset, self.category)
        self.result_page.set_input_image(image_path)

        pred = "Defective" if score > self.threshold else "Good"
        status = "Fail" if pred == "Defective" else "Pass"
        self.result_page.set_results(score=score, prediction=pred, status=status)
        self.result_page.set_heatmap(heatmap)  # placeholder only

        self.stack.setCurrentWidget(self.result_page)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PCBApp()
    window.show()
    sys.exit(app.exec())
