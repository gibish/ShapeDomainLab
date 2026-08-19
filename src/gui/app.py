import sys
from html import escape
from io import BytesIO
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    SUPPORTED_COLOR_MODES,
    SUPPORTED_FILL_MODES,
    SUPPORTED_IMAGE_SIZES,
    DatasetConfig,
    ShapeRenderConfig,
)
from src.__version__ import __version__
from src.core.config_io import load_config_from_json, save_config_template
from src.core.generator import SUPPORTED_IMAGE_FORMATS, clear_dataset_dir, validate_generated_dataset_dir
from src.core.paths import resolve_output_dir
from src.core.preview import generate_preview_images
from src.core.summary import DatasetSummary, dataset_summary, estimate_dataset_size
from src.gui.worker import GenerationWorker


APP_ICON_PATH = Path(__file__).resolve().parents[2] / "assets" / "ShapeDomainLab_Icon.png"
CHECKMARK_PATH = Path(__file__).resolve().parents[2] / "assets" / "checkmark.svg"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ShapeDomainLab")
        self.resize(1480, 780)
        if APP_ICON_PATH.is_file():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self._worker: GenerationWorker | None = None
        self._last_output_dir: Path | None = None
        self._tooltips_enabled = True
        self._tooltip_texts: dict[QWidget, str] = {}

        self._build_menu_bar()
        self._apply_style()

        root = QWidget()
        self.setCentralWidget(root)
        root.setObjectName("Workspace")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 12)
        root_layout.setSpacing(12)

        content_layout = QGridLayout()
        content_layout.setHorizontalSpacing(12)
        content_layout.setVerticalSpacing(12)
        content_layout.setColumnStretch(0, 2)
        content_layout.setColumnStretch(1, 1)
        content_layout.setColumnStretch(2, 1)
        content_layout.setColumnStretch(3, 1)
        content_layout.setRowStretch(1, 1)

        content_layout.addWidget(self._build_global_group(), 0, 0)
        content_layout.addWidget(self._build_split_group(), 0, 1)
        content_layout.addWidget(self._build_classes_group(), 0, 2)
        content_layout.addWidget(self._build_preview_group(), 0, 3, 2, 1)
        content_layout.addWidget(self._build_render_group(), 1, 0, 1, 3)
        bottom_panels = QWidget()
        bottom_panels_layout = QHBoxLayout(bottom_panels)
        bottom_panels_layout.setContentsMargins(0, 0, 0, 0)
        bottom_panels_layout.setSpacing(12)
        bottom_panels_layout.addWidget(self._build_summary_group(), 2)
        bottom_panels_layout.addWidget(self._build_log_group(), 3)
        content_layout.addWidget(bottom_panels, 2, 0, 1, 4)

        root_layout.addLayout(content_layout, 1)
        root_layout.addLayout(self._build_controls())
        self._load_preview_if_dataset()

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        self.load_config_action = QAction("Load Config...", self)
        self.save_config_action = QAction("Save Config...", self)
        exit_action = QAction("Exit", self)
        self.load_config_action.triggered.connect(self._load_config)
        self.save_config_action.triggered.connect(self._save_config)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(self.load_config_action)
        file_menu.addAction(self.save_config_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        dataset_menu = self.menuBar().addMenu("Dataset")
        self.generate_action = QAction("Generate", self)
        self.cancel_action = QAction("Cancel", self)
        self.clear_action = QAction("Clear Dataset", self)
        self.cancel_action.setEnabled(False)
        self.generate_action.triggered.connect(self._start_generation)
        self.cancel_action.triggered.connect(self._cancel_generation)
        self.clear_action.triggered.connect(self._clear_dataset)
        dataset_menu.addAction(self.generate_action)
        dataset_menu.addAction(self.cancel_action)
        dataset_menu.addSeparator()
        dataset_menu.addAction(self.clear_action)

        tools_menu = self.menuBar().addMenu("Tools")
        self.refresh_preview_action = QAction("Refresh Preview", self)
        self.summary_action = QAction("Dataset Summary", self)
        self.estimate_action = QAction("Estimate Size", self)
        self.refresh_preview_action.triggered.connect(self._refresh_preview)
        self.summary_action.triggered.connect(self._show_dataset_summary)
        self.estimate_action.triggered.connect(self._show_estimate)
        tools_menu.addAction(self.refresh_preview_action)
        tools_menu.addAction(self.summary_action)
        tools_menu.addAction(self.estimate_action)

        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About ShapeDomainLab", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("AppHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(14)

        icon_label = QLabel()
        icon_label.setObjectName("AppIcon")
        icon_label.setFixedSize(72, 72)
        icon_label.setAlignment(Qt.AlignCenter)
        if APP_ICON_PATH.is_file():
            icon = QPixmap(str(APP_ICON_PATH))
            icon_label.setPixmap(icon.scaled(68, 68, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            icon_label.setText("SSC")

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        title = QLabel("ShapeDomainLab")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Generator of simple geometric shape datasets for computer vision experiments")
        subtitle.setObjectName("AppSubtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        layout.addWidget(icon_label)
        layout.addLayout(title_layout, 1)
        return header

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f7f9fc;
                color: #1f2933;
                font-size: 10pt;
            }
            QLabel {
                background: transparent;
            }
            QGroupBox QLabel, #FormSection QLabel {
                background: transparent;
            }
            QCheckBox {
                background: transparent;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #b8c4d0;
                border-radius: 3px;
                background: transparent;
            }
            QCheckBox::indicator:checked {
                background: #0b63e5;
                border: 1px solid #0b63e5;
                image: url("__CHECKMARK_PATH__");
            }
            #Workspace {
                background: #f7f9fc;
            }
            QMenuBar {
                background: #ffffff;
                border-bottom: 1px solid #d9e2ec;
                padding: 3px;
            }
            QMenuBar::item:selected, QMenu::item:selected {
                background: #e3f2fd;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 7px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #cbd6e2;
                border-radius: 5px;
                padding: 7px 10px;
            }
            QPushButton:hover {
                background: #eef6ff;
            }
            QPushButton:disabled {
                color: #8a97a5;
                background: #edf1f5;
                border-color: #d3dbe3;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background: #ffffff;
                border: 1px solid #c8d4df;
                border-radius: 4px;
                padding: 3px;
            }
            QProgressBar {
                border: 1px solid #c8d4df;
                border-radius: 4px;
                text-align: center;
                background: #ffffff;
            }
            QProgressBar::chunk {
                background: #2f80c0;
                border-radius: 3px;
            }
            #AppHeader {
                background: #ffffff;
                border: 1px solid #c8d4df;
                border-radius: 8px;
            }
            #AppIcon {
                background: #eaf6ff;
                border: 1px solid #bdd5ea;
                border-radius: 8px;
            }
            #AppTitle {
                font-size: 20pt;
                font-weight: 700;
                color: #0f3d66;
            }
            #AppSubtitle {
                color: #536878;
                font-size: 10pt;
            }
            #FormSection {
                background: #fbfdff;
                border: 1px solid #e3ebf3;
                border-radius: 6px;
            }
            #FormSectionTitle {
                background: transparent;
                color: #0f3d66;
                font-weight: 700;
                font-size: 10pt;
            }
            QWidget#TransparentFieldRow, QLabel#TransparentFieldLabel {
                background: transparent;
            }
            QPushButton#PrimaryButton {
                background: #0b63e5;
                color: #ffffff;
                border: 1px solid #0a58cc;
                font-weight: 600;
                padding: 10px 18px;
            }
            QPushButton#PrimaryButton:hover {
                background: #095bd4;
            }
            QPushButton#BottomButton {
                padding: 10px 18px;
                font-weight: 500;
            }
            """.replace("__CHECKMARK_PATH__", CHECKMARK_PATH.as_posix())
        )

    def _build_global_group(self) -> QGroupBox:
        group = QGroupBox("1. Dataset Parameters")
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setVerticalSpacing(12)

        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        self.output_edit = QLineEdit("datasets/shapes")
        self.browse_button = QPushButton("Browse")
        output_tooltip = "Dataset output directory. Relative paths are resolved from the project root."
        self._set_tooltip(self.output_edit, output_tooltip)
        self._set_tooltip(self.browse_button, "Choose the dataset output directory.")
        self.browse_button.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_edit)
        output_row.addWidget(self.browse_button)
        output_label = QLabel("Output")
        output_label.setObjectName("TransparentFieldLabel")
        self._set_tooltip(output_label, output_tooltip)
        layout.addRow(output_label, output_row)

        self.image_size_combo = QComboBox()
        self.image_size_combo.addItems(str(size) for size in SUPPORTED_IMAGE_SIZES)
        self.image_size_combo.setCurrentText(str(DatasetConfig().image_size))
        self.image_size_combo.currentTextChanged.connect(self._update_center_offset_limits)
        self.train_count_spin = self._spinbox(1, 1_000_000, 10)
        self.val_count_spin = self._spinbox(1, 1_000_000, 10)
        self.test_count_spin = self._spinbox(1, 1_000_000, 10)
        self.seed_spin = self._spinbox(0, 2_147_483_647, 12345)
        self.save_masks_check = QCheckBox("Save masks")
        self._set_tooltip(
            self.save_masks_check,
            "Save one binary PNG segmentation mask per image in the masks/ directory.",
        )
        self.format_combo = QComboBox()
        self.format_combo.addItems(SUPPORTED_IMAGE_FORMATS)
        self.format_combo.setCurrentText("png")

        size_format_widget = QWidget()
        size_format_widget.setObjectName("TransparentFieldRow")
        size_format_layout = QGridLayout(size_format_widget)
        size_format_layout.setContentsMargins(0, 0, 0, 0)
        size_format_layout.setHorizontalSpacing(10)
        size_format_layout.setColumnStretch(1, 1)
        size_format_layout.setColumnStretch(3, 1)
        image_size_label = QLabel("Image size")
        format_label = QLabel("Format")
        image_size_label.setObjectName("TransparentFieldLabel")
        format_label.setObjectName("TransparentFieldLabel")
        image_size_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        format_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        image_size_tooltip = "Square image size in pixels. Available values: 28, 32, 64, 128, 224, 256."
        format_tooltip = "Image file format. Available values: png, jpg, gif."
        self._set_tooltip(image_size_label, image_size_tooltip)
        self._set_tooltip(self.image_size_combo, image_size_tooltip)
        self._set_tooltip(format_label, format_tooltip)
        self._set_tooltip(self.format_combo, format_tooltip)
        size_format_layout.addWidget(image_size_label, 0, 0)
        size_format_layout.addWidget(self.image_size_combo, 0, 1)
        size_format_layout.addWidget(format_label, 0, 2)
        size_format_layout.addWidget(self.format_combo, 0, 3)
        layout.addRow(size_format_widget)
        seed_widget = QWidget()
        seed_widget.setObjectName("TransparentFieldRow")
        seed_layout = QHBoxLayout(seed_widget)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(12)
        seed_layout.addWidget(self.seed_spin)
        seed_layout.addWidget(self.save_masks_check)
        self._add_form_row(
            layout,
            "Seed",
            seed_widget,
            "Random seed for reproducible generation. Range: 0 to 2,147,483,647.",
        )
        return group

    def _build_split_group(self) -> QGroupBox:
        group = QGroupBox("2. Dataset Split")
        group.setMaximumWidth(430)
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setVerticalSpacing(12)

        self.automatic_split_check = QCheckBox("Automatic")
        self.manual_split_check = QCheckBox("Manual")
        self.automatic_split_check.setStyleSheet("background: transparent;")
        self.manual_split_check.setStyleSheet("background: transparent;")
        self.automatic_split_check.setChecked(True)
        self.automatic_split_check.toggled.connect(lambda checked: self._set_split_mode(automatic=checked))
        self.manual_split_check.toggled.connect(lambda checked: self._set_split_mode(automatic=not checked))
        mode_widget = QWidget()
        mode_widget.setObjectName("TransparentFieldRow")
        mode_widget.setStyleSheet("background: transparent;")
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(12)
        mode_layout.addWidget(self.automatic_split_check)
        mode_layout.addWidget(self.manual_split_check)
        mode_layout.addStretch(1)
        mode_tooltip = "Choose Automatic to calculate split sizes from a total dataset count, or Manual to set each subset count yourself."
        mode_label = QLabel("Mode")
        mode_label.setObjectName("TransparentFieldLabel")
        self._set_tooltip(mode_label, mode_tooltip)
        self._set_tooltip(self.automatic_split_check, mode_tooltip)
        self._set_tooltip(self.manual_split_check, mode_tooltip)
        layout.addRow(mode_label, mode_widget)

        self.dataset_count_spin = self._spinbox(1, 1_000_000, 100)
        self.split_ratio_combo = QComboBox()
        self.split_ratio_combo.addItems(["70 / 15 / 15", "80 / 10 / 10"])
        self.split_ratio_combo.setEditable(True)
        self.split_ratio_combo.lineEdit().setReadOnly(True)
        self.dataset_count_spin.valueChanged.connect(self._update_split_counts)
        self.split_ratio_combo.currentTextChanged.connect(self._update_split_counts)
        layout.addRow(
            self._split_summary_row(
                "Dataset count",
                self.dataset_count_spin,
                "Total number of images across all subsets in Automatic mode.",
                "Ratio",
                self.split_ratio_combo,
                "Automatic split ratio: train / val / test.",
            )
        )

        layout.addRow(
            self._triple_field_widget(
                (
                    ("Train", self.train_count_spin, "Train subset image count. In Manual mode, enter it directly. In Automatic mode, it is calculated."),
                    ("Validation", self.val_count_spin, "Validation subset image count. In Manual mode, enter it directly. In Automatic mode, it is calculated."),
                    ("Test", self.test_count_spin, "Test subset image count. In Manual mode, enter it directly. In Automatic mode, it is calculated."),
                )
            )
        )
        self._set_split_mode(automatic=True)
        self._update_split_counts()
        for field in (self.train_count_spin, self.val_count_spin, self.test_count_spin):
            field.valueChanged.connect(self._update_manual_split_summary)
        return group

    def _build_classes_group(self) -> QGroupBox:
        group = QGroupBox("3. Shape Classes")
        layout = QVBoxLayout(group)
        self.class_checks: dict[str, QCheckBox] = {}
        for class_name in ("circle", "octagon", "rectangle", "square", "triangle"):
            checkbox = QCheckBox(class_name)
            self._set_tooltip(checkbox, f"Generate images for the `{class_name}` class.")
            checkbox.setChecked(True)
            self.class_checks[class_name] = checkbox
            layout.addWidget(checkbox)
        layout.addStretch(1)
        return group

    def _build_render_group(self) -> QGroupBox:
        group = QGroupBox("4. Shape Parameters")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(SUPPORTED_COLOR_MODES)
        self.color_mode_combo.setCurrentText("grayscale")
        self.background_min_spin = self._spinbox(0, 255, 150)
        self.background_max_spin = self._spinbox(1, 256, 200)
        self.foreground_min_spin = self._spinbox(0, 255, 0)
        self.foreground_max_spin = self._spinbox(1, 256, 50)
        self.background_color = (180, 180, 180)
        self.foreground_color = (0, 0, 0)
        self.background_color_button = self._color_button("Background color", self.background_color)
        self.foreground_color_button = self._color_button("Shape color", self.foreground_color)
        self.line_min_spin = self._spinbox(1, 64, 1)
        self.line_max_spin = self._spinbox(2, 65, 4)
        self.fill_mode_combo = QComboBox()
        self.fill_mode_combo.addItems(SUPPORTED_FILL_MODES)
        self.fill_mode_combo.setCurrentText("outline")
        self.rotation_min_spin = self._double_spinbox(-180.0, 180.0, 0.0, 1.0, 1)
        self.rotation_max_spin = self._double_spinbox(-180.0, 180.0, 0.0, 1.0, 1)
        self.scale_min_spin = self._double_spinbox(0.1, 2.0, 1.0, 0.1, 2)
        self.scale_max_spin = self._double_spinbox(0.1, 2.0, 1.0, 0.1, 2)
        self.center_offset_min_spin = self._double_spinbox(0.0, 512.0, 0.0, 1.0, 1)
        self.center_offset_max_spin = self._double_spinbox(0.0, 512.0, 0.0, 1.0, 1)
        self._update_center_offset_limits()
        self.edge_clipping_check = QCheckBox("Edge clipping")
        self.min_visible_ratio_spin = self._double_spinbox(0.6, 1.0, 0.6, 0.05, 2)
        self.min_visible_ratio_spin.setEnabled(False)
        self._set_tooltip(
            self.edge_clipping_check,
            "Allow a shape to be partially clipped by one image edge.",
        )
        self._set_tooltip(
            self.min_visible_ratio_spin,
            "Minimum visible part of a clipped shape. Range: 0.60 to 1.00.",
        )
        self.edge_clipping_check.toggled.connect(self.min_visible_ratio_spin.setEnabled)
        self.noise_check = QCheckBox("Noise")
        self.noise_level_spin = self._spinbox(0, 128, 20)
        self.noise_level_spin.setEnabled(False)
        self._set_tooltip(self.noise_check, "Enable uniform random noise on the image background.")
        self._set_tooltip(self.noise_level_spin, "Uniform background noise strength. Range: 0 to 128.")
        self.noise_check.toggled.connect(self.noise_level_spin.setEnabled)
        self.margin_spin = self._spinbox(0, 128, 2)
        self.color_mode_combo.currentTextChanged.connect(self._set_color_controls_enabled)
        self.background_color_button.clicked.connect(
            lambda: self._choose_color("Background Color", "background_color", self.background_color_button)
        )
        self.foreground_color_button.clicked.connect(
            lambda: self._choose_color("Shape Color", "foreground_color", self.foreground_color_button)
        )
        self.color_controls_stack = QStackedWidget()
        grayscale_controls = QWidget()
        grayscale_layout = QVBoxLayout(grayscale_controls)
        grayscale_layout.setContentsMargins(0, 0, 0, 0)
        grayscale_layout.setSpacing(6)
        grayscale_layout.addWidget(
            self._paired_field_widget(
                "Background min",
                self.background_min_spin,
                "Minimum background grayscale intensity. Range: 0 to 255; must be smaller than Background max.",
                "max",
                self.background_max_spin,
                "Exclusive maximum background grayscale intensity. Range: 1 to 256; must be larger than Background min.",
            )
        )
        grayscale_layout.addWidget(
            self._paired_field_widget(
                "Foreground min",
                self.foreground_min_spin,
                "Minimum shape grayscale intensity. Range: 0 to 255; must be smaller than Foreground max.",
                "max",
                self.foreground_max_spin,
                "Exclusive maximum shape grayscale intensity. Range: 1 to 256; must be larger than Foreground min.",
            )
        )
        grayscale_layout.addStretch(1)
        rgb_controls = QWidget()
        rgb_layout = QFormLayout(rgb_controls)
        rgb_layout.setContentsMargins(0, 0, 0, 0)
        rgb_layout.setHorizontalSpacing(12)
        rgb_layout.setVerticalSpacing(6)
        self._add_form_row(
            rgb_layout,
            "Background color",
            self.background_color_button,
            "RGB background color used in rgb mode.",
        )
        self._add_form_row(
            rgb_layout,
            "Shape color",
            self.foreground_color_button,
            "RGB shape color used in rgb mode.",
        )
        self.color_controls_stack.addWidget(grayscale_controls)
        self.color_controls_stack.addWidget(rgb_controls)

        main_sections = QGridLayout()
        main_sections.setHorizontalSpacing(10)
        main_sections.setVerticalSpacing(10)
        main_sections.setColumnStretch(0, 1)
        main_sections.setColumnStretch(1, 1)
        main_sections.setColumnMinimumWidth(0, 490)
        main_sections.setColumnMinimumWidth(1, 490)
        layout.addLayout(main_sections)
        appearance_layout = self._section_form(main_sections, "Appearance", 0, 0, 2, 1)
        geometry_layout = self._section_form(main_sections, "Geometry", 0, 1)
        effects_layout = self._section_form(main_sections, "Effects", 1, 1)

        self._add_form_row(
            appearance_layout,
            "Color mode",
            self.color_mode_combo,
            "Image color mode: grayscale uses intensity ranges; rgb uses fixed background and shape RGB colors.",
        )
        appearance_layout.addRow(self.color_controls_stack)
        appearance_layout.addRow(
            self._paired_field_widget(
                "Line width min",
                self.line_min_spin,
                "Minimum outline width in pixels. Range: 1 to 64; must be smaller than Line width max.",
                "max",
                self.line_max_spin,
                "Exclusive maximum outline width in pixels. Range: 2 to 65; must be larger than Line width min.",
            )
        )
        self._add_form_row(
            appearance_layout,
            "Fill mode",
            self.fill_mode_combo,
            "Shape drawing mode: outline = contour only; filled = fill only; outline_fill = fill plus contour.",
        )
        geometry_layout.addRow(
            self._paired_field_widget(
                "Rotation min",
                self.rotation_min_spin,
                "Minimum random rotation angle in degrees. Range: -180.0 to 180.0.",
                "max",
                self.rotation_max_spin,
                "Maximum random rotation angle in degrees. Range: -180.0 to 180.0; must be at least Rotation min.",
            )
        )
        geometry_layout.addRow(
            self._paired_field_widget(
                "Scale min",
                self.scale_min_spin,
                "Minimum random shape scale factor. Range: 0.10 to 2.00.",
                "max",
                self.scale_max_spin,
                "Maximum random shape scale factor. Range: 0.10 to 2.00; must be at least Scale min.",
            )
        )
        geometry_layout.addRow(
            self._paired_field_widget(
                "Center offset min",
                self.center_offset_min_spin,
                "Minimum random center offset from image center in pixels. Maximum is half the image size.",
                "max",
                self.center_offset_max_spin,
                "Maximum random center offset from image center in pixels. Maximum is half the image size.",
            )
        )
        self._add_form_row(
            geometry_layout,
            "Margin",
            self.margin_spin,
            "Minimum shape distance from the image edge in pixels. Range: 0 to 128.",
        )
        effects_row = QWidget()
        effects_row_layout = QGridLayout(effects_row)
        effects_row_layout.setContentsMargins(0, 0, 0, 0)
        effects_row_layout.setHorizontalSpacing(10)
        effects_row_layout.setColumnStretch(1, 1)
        effects_row_layout.setColumnStretch(3, 1)
        effects_row_layout.addWidget(self.edge_clipping_check, 0, 0)
        effects_row_layout.addWidget(self.min_visible_ratio_spin, 0, 1)
        effects_row_layout.addWidget(self.noise_check, 0, 2)
        effects_row_layout.addWidget(self.noise_level_spin, 0, 3)
        effects_layout.addRow(effects_row)
        self._set_color_controls_enabled(self.color_mode_combo.currentText())
        return group

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("Preview")
        outer_layout = QVBoxLayout(group)
        layout = QGridLayout()
        self.preview_labels: list[QLabel] = []
        for index in range(9):
            label = QLabel("No image")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(120, 120)
            label.setObjectName("PreviewTile")
            label.setStyleSheet("QLabel#PreviewTile { border: 1px solid #dbe4ee; border-radius: 6px; background: #fbfcfe; }")
            self.preview_labels.append(label)
            layout.addWidget(label, index // 3, index % 3)
        outer_layout.addLayout(layout)
        return group

    def _build_summary_group(self) -> QGroupBox:
        group = QGroupBox("5. Options and Dataset Summary")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)
        self.summary_edit = QTextEdit()
        self.summary_edit.setReadOnly(True)
        self._set_text_edit_rows(self.summary_edit, 5)
        self.summary_edit.setPlainText("No dataset summary loaded.")
        self.open_summary_button = QPushButton("Open Summary")
        self._set_tooltip(self.open_summary_button, "Open the full summary in a larger window.")
        self.open_summary_button.clicked.connect(self._show_full_summary)
        layout.addWidget(self.summary_edit, 1)
        layout.addWidget(self.open_summary_button)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Log")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self._set_text_edit_rows(self.log_edit, 5)
        self.open_log_button = QPushButton("Open Log")
        self._set_tooltip(self.open_log_button, "Open the full log in a larger window.")
        self.open_log_button.clicked.connect(self._show_full_log)
        layout.addWidget(self.log_edit, 1)
        layout.addWidget(self.open_log_button)
        return group

    def _set_text_edit_rows(self, text_edit: QTextEdit, rows: int) -> None:
        line_height = text_edit.fontMetrics().lineSpacing()
        text_edit.setMinimumHeight(line_height * rows + 18)

    def _show_full_summary(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Full Summary")
        dialog.resize(900, 600)
        layout = QVBoxLayout(dialog)
        summary_view = QTextEdit()
        summary_view.setReadOnly(True)
        summary_view.setPlainText(self.summary_edit.toPlainText())
        layout.addWidget(summary_view)
        dialog.exec()

    def _show_full_log(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Full Log")
        dialog.resize(900, 600)
        layout = QVBoxLayout(dialog)
        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setPlainText(self.log_edit.toPlainText())
        layout.addWidget(log_view)
        dialog.exec()

    def _build_controls(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate")
        self.bottom_preview_button = QPushButton("Preview")
        self.summary_button = QPushButton("Summary")
        self.estimate_button = QPushButton("Estimate")
        self.save_config_button = QPushButton("Save Config")
        self.load_config_button = QPushButton("Load Config")
        self.cancel_button = QPushButton("Cancel")
        self.clear_button = QPushButton("Clear Dataset")
        self.cancel_button.hide()
        self.clear_button.hide()
        self.generate_button.setObjectName("PrimaryButton")
        self.bottom_preview_button.setObjectName("BottomButton")
        self.summary_button.setObjectName("BottomButton")
        self.estimate_button.setObjectName("BottomButton")
        self.save_config_button.setObjectName("BottomButton")
        self.load_config_button.setObjectName("BottomButton")
        self.cancel_button.setObjectName("BottomButton")
        self.clear_button.setObjectName("BottomButton")
        self.tooltips_check = QCheckBox("Tooltips")
        self.tooltips_check.setChecked(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self._set_tooltip(self.generate_button, "Generate the dataset with the selected parameters.")
        self._set_tooltip(self.cancel_button, "Request cancellation of the running generation job.")
        self._set_tooltip(self.clear_button, "Clear the selected generated dataset after confirmation.")
        self._set_tooltip(self.bottom_preview_button, "Render sample images from the current parameters.")
        self._set_tooltip(self.summary_button, "Show a summary for the selected output dataset directory.")
        self._set_tooltip(self.estimate_button, "Estimate image count and output size for the current settings.")
        self._set_tooltip(self.save_config_button, "Save the current generation settings to a JSON config file.")
        self._set_tooltip(self.load_config_button, "Load generation settings from a JSON config file.")
        self._set_tooltip(self.tooltips_check, "Show or hide hover tooltips in the interface.")
        self._set_tooltip(self.progress_bar, "Generation progress for the current job.")
        self._set_tooltip(self.status_label, "Current generation or dataset status.")

        self.generate_button.clicked.connect(self._start_generation)
        self.bottom_preview_button.clicked.connect(self._refresh_preview)
        self.summary_button.clicked.connect(self._show_dataset_summary)
        self.estimate_button.clicked.connect(self._show_estimate)
        self.save_config_button.clicked.connect(self._save_config)
        self.load_config_button.clicked.connect(self._load_config)
        self.cancel_button.clicked.connect(self._cancel_generation)
        self.clear_button.clicked.connect(self._clear_dataset)
        self.tooltips_check.toggled.connect(self._set_tooltips_enabled)

        layout.addWidget(self.generate_button)
        layout.addWidget(self.bottom_preview_button)
        layout.addWidget(self.summary_button)
        layout.addWidget(self.estimate_button)
        layout.addWidget(self.save_config_button)
        layout.addWidget(self.load_config_button)
        layout.addWidget(self.tooltips_check)
        layout.addWidget(self.progress_bar, 1)
        layout.addWidget(self.status_label)
        return layout

    def _spinbox(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        return spinbox

    def _double_spinbox(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(decimals)
        return spinbox

    def _color_button(self, label: str, color: tuple[int, int, int]) -> QPushButton:
        button = QPushButton()
        self._update_color_button(button, label, color)
        return button

    def _choose_color(self, title: str, attribute: str, button: QPushButton) -> None:
        current = QColor(*getattr(self, attribute))
        dialog = QColorDialog(current, self)
        dialog.setWindowTitle(title)
        dialog.setOption(QColorDialog.DontUseNativeDialog, True)
        if dialog.exec() != QColorDialog.Accepted:
            return
        color = dialog.selectedColor()
        if not color.isValid():
            return
        selected = (color.red(), color.green(), color.blue())
        setattr(self, attribute, selected)
        self._update_color_button(button, title, selected)

    def _update_color_button(self, button: QPushButton, label: str, color: tuple[int, int, int]) -> None:
        red, green, blue = color
        button.setText(f"{label}: #{red:02X}{green:02X}{blue:02X}")
        button.setStyleSheet(
            "QPushButton {"
            f"background-color: rgb({red}, {green}, {blue});"
            f"color: {'#ffffff' if red * 0.299 + green * 0.587 + blue * 0.114 < 140 else '#1b2430'};"
            "border: 1px solid #9aa8b7;"
            "padding: 5px 10px;"
            "}"
        )

    def _paired_field_widget(
        self,
        left_label: str,
        left_widget: QWidget,
        left_tooltip: str,
        right_label: str,
        right_widget: QWidget,
        right_tooltip: str,
    ) -> QWidget:
        widget = QWidget()
        widget.setObjectName("TransparentFieldRow")
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        left_label_widget = QLabel(left_label)
        right_label_widget = QLabel(right_label)
        left_label_widget.setObjectName("TransparentFieldLabel")
        right_label_widget.setObjectName("TransparentFieldLabel")
        # Keep the second (max) control in a fixed column across paired rows.
        left_label_widget.setFixedWidth(125)
        if right_label == "max":
            right_label_widget.setFixedWidth(32)
        left_label_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        right_label_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._set_tooltip(left_label_widget, left_tooltip)
        self._set_tooltip(left_widget, left_tooltip)
        self._set_tooltip(right_label_widget, right_tooltip)
        self._set_tooltip(right_widget, right_tooltip)
        left_widget.setSizePolicy(QSizePolicy.Expanding, left_widget.sizePolicy().verticalPolicy())
        right_widget.setSizePolicy(QSizePolicy.Expanding, right_widget.sizePolicy().verticalPolicy())
        layout.addWidget(left_label_widget, 0, 0)
        layout.addWidget(left_widget, 0, 1)
        layout.addWidget(right_label_widget, 0, 2)
        layout.addWidget(right_widget, 0, 3)
        return widget

    def _triple_field_widget(self, fields: tuple[tuple[str, QWidget, str], ...]) -> QWidget:
        widget = QWidget()
        widget.setObjectName("TransparentFieldRow")
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        for index, (label, field, tooltip) in enumerate(fields):
            label_widget = QLabel(label)
            label_widget.setObjectName("TransparentFieldLabel")
            label_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._set_tooltip(label_widget, tooltip)
            self._set_tooltip(field, tooltip)
            field.setSizePolicy(QSizePolicy.Expanding, field.sizePolicy().verticalPolicy())
            column = index * 2
            layout.addWidget(label_widget, 0, column)
            layout.addWidget(field, 0, column + 1)
            layout.setColumnStretch(column + 1, 1)
        return widget

    def _split_summary_row(
        self,
        left_label: str,
        left_widget: QWidget,
        left_tooltip: str,
        right_label: str,
        right_widget: QWidget,
        right_tooltip: str,
    ) -> QWidget:
        widget = QWidget()
        widget.setObjectName("TransparentFieldRow")
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        left_label_widget = QLabel(left_label)
        right_label_widget = QLabel(right_label)
        for label_widget, tooltip in ((left_label_widget, left_tooltip), (right_label_widget, right_tooltip)):
            label_widget.setObjectName("TransparentFieldLabel")
            label_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._set_tooltip(label_widget, tooltip)
        self._set_tooltip(left_widget, left_tooltip)
        self._set_tooltip(right_widget, right_tooltip)
        right_widget.setSizePolicy(QSizePolicy.Expanding, right_widget.sizePolicy().verticalPolicy())
        layout.addWidget(left_label_widget, 0, 0)
        layout.addWidget(left_widget, 0, 1)
        layout.addWidget(right_label_widget, 0, 2)
        layout.addWidget(right_widget, 0, 3)
        layout.setColumnStretch(3, 1)
        return widget

    def _set_split_mode(self, automatic: bool) -> None:
        self.automatic_split_check.blockSignals(True)
        self.manual_split_check.blockSignals(True)
        self.automatic_split_check.setChecked(automatic)
        self.manual_split_check.setChecked(not automatic)
        self.automatic_split_check.blockSignals(False)
        self.manual_split_check.blockSignals(False)
        self.dataset_count_spin.setEnabled(automatic)
        self.split_ratio_combo.setEnabled(automatic)
        self.train_count_spin.setEnabled(not automatic)
        self.val_count_spin.setEnabled(not automatic)
        self.test_count_spin.setEnabled(not automatic)
        if automatic:
            if not self.split_ratio_combo.currentText().startswith(("70", "80")):
                self.split_ratio_combo.setCurrentText("70 / 15 / 15")
            self._update_split_counts()
        else:
            self._update_manual_split_summary()

    def _selected_split_ratio(self) -> tuple[int, int, int]:
        return (70, 15, 15) if self.split_ratio_combo.currentText().startswith("70") else (80, 10, 10)

    def _update_split_counts(self) -> None:
        if not self.automatic_split_check.isChecked():
            return
        total = self.dataset_count_spin.value()
        train_ratio, val_ratio, test_ratio = self._selected_split_ratio()
        val_count = int(total * val_ratio / 100)
        test_count = val_count
        train_count = max(1, total - val_count - test_count)
        self.train_count_spin.setValue(train_count)
        self.val_count_spin.setValue(val_count)
        self.test_count_spin.setValue(test_count)

    def _update_manual_split_summary(self) -> None:
        if self.automatic_split_check.isChecked():
            return
        train = self.train_count_spin.value()
        val = self.val_count_spin.value()
        test = self.test_count_spin.value()
        total = train + val + test
        self.dataset_count_spin.blockSignals(True)
        self.dataset_count_spin.setValue(total)
        self.dataset_count_spin.blockSignals(False)
        if total:
            self.split_ratio_combo.blockSignals(True)
            self.split_ratio_combo.setCurrentText(
                f"{train / total * 100:.1f} / {val / total * 100:.1f} / {test / total * 100:.1f}"
            )
            self.split_ratio_combo.blockSignals(False)

    def _section_form(
        self,
        parent_layout: QLayout,
        title: str,
        row: int | None = None,
        column: int | None = None,
        row_span: int = 1,
        column_span: int = 1,
    ) -> QFormLayout:
        section = QFrame()
        section.setObjectName("FormSection")
        section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(10, 8, 10, 8)
        section_layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("FormSectionTitle")
        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form_layout.setLabelAlignment(Qt.AlignLeft)
        form_layout.setFormAlignment(Qt.AlignTop)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(6)
        section_layout.addWidget(title_label)
        section_layout.addLayout(form_layout)
        if isinstance(parent_layout, QGridLayout):
            if row is None or column is None:
                raise ValueError("Grid section placement requires row and column.")
            parent_layout.addWidget(section, row, column, row_span, column_span)
        else:
            parent_layout.addWidget(section)
        if isinstance(parent_layout, QHBoxLayout):
            parent_layout.setStretch(parent_layout.count() - 1, 1)
        return form_layout

    def _set_color_controls_enabled(self, mode: str) -> None:
        grayscale_enabled = mode == "grayscale"
        self.color_controls_stack.setCurrentIndex(0 if grayscale_enabled else 1)
        for spinbox in (
            self.background_min_spin,
            self.background_max_spin,
            self.foreground_min_spin,
            self.foreground_max_spin,
        ):
            spinbox.setEnabled(grayscale_enabled)

    def _update_center_offset_limits(self) -> None:
        if not hasattr(self, "center_offset_min_spin") or not hasattr(self, "center_offset_max_spin"):
            return
        maximum = int(self.image_size_combo.currentText()) / 2
        for spinbox in (self.center_offset_min_spin, self.center_offset_max_spin):
            spinbox.setMaximum(maximum)
            if spinbox.value() > maximum:
                spinbox.setValue(maximum)

    def _set_tooltip(self, widget: QWidget, tooltip: str) -> None:
        if tooltip:
            self._tooltip_texts[widget] = tooltip
        else:
            self._tooltip_texts.pop(widget, None)
        widget.setToolTip(tooltip if self._tooltips_enabled else "")

    def _set_tooltips_enabled(self, enabled: bool) -> None:
        self._tooltips_enabled = enabled
        for widget, tooltip in self._tooltip_texts.items():
            widget.setToolTip(tooltip if enabled else "")

    def _add_form_row(
        self,
        layout: QFormLayout,
        label: str,
        widget: QWidget,
        tooltip: str,
    ) -> None:
        label_widget = QLabel(label)
        label_widget.setObjectName("TransparentFieldLabel")
        self._set_tooltip(label_widget, tooltip)
        self._set_tooltip(widget, tooltip)
        widget.setSizePolicy(QSizePolicy.Expanding, widget.sizePolicy().verticalPolicy())
        layout.addRow(label_widget, widget)

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_edit.text())
        if directory:
            self.output_edit.setText(directory)
            self._load_preview_if_dataset()
            self._show_dataset_summary()

    def _load_config(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Generation Config",
            self.output_edit.text(),
            "JSON files (*.json);;All files (*)",
        )
        if not filename:
            return

        try:
            config = load_config_from_json(filename)
        except (OSError, ValueError) as exc:
            self._show_config_load_error(str(exc))
            self._append_log(f"Config load failed: {exc}")
            return

        self._apply_config(config)
        self.status_label.setText("Config loaded")
        self._append_log(f"Config loaded: {filename}")
        self._refresh_preview()

    def _show_config_load_error(self, message: str) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle("Load Config Failed")
        dialog.setTextFormat(Qt.RichText)
        dialog.setText(
            "<span style='color:#b00020; font-weight:600;'>"
            "Selected file is not a valid ShapeDomainLab config."
            "</span>"
        )
        dialog.setInformativeText(escape(message))
        dialog.exec()

    def _save_config(self) -> None:
        try:
            config = self._build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Parameters", str(exc))
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generation Config",
            "generation_config.json",
            "JSON files (*.json);;All files (*)",
        )
        if not filename:
            return

        try:
            saved_path = save_config_template(filename, config)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save Config Failed", str(exc))
            self._append_log(f"Config save failed: {exc}")
            return

        self.status_label.setText("Config saved")
        self._append_log(f"Config saved: {saved_path}")

    def _show_dataset_summary(self) -> None:
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "Invalid Output", "Output directory cannot be empty.")
            return

        summary = dataset_summary(resolve_output_dir(Path(output_text)))
        self.summary_edit.setPlainText(self._format_summary(summary))
        self.status_label.setText("Summary updated")
        self._append_log(f"Dataset summary updated: {summary.output_dir}")

    def _show_estimate(self) -> None:
        try:
            config = self._build_config()
            estimate = estimate_dataset_size(config)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Parameters", str(exc))
            return

        lines = [
            f"Estimated images: {estimate.total_images}",
            f"Sample images used: {estimate.sample_count}",
            f"Estimated image data: {self._format_bytes(estimate.estimated_image_bytes)}",
            f"Estimated service files: {self._format_bytes(estimate.estimated_service_bytes)}",
            f"Estimated total: {self._format_bytes(estimate.estimated_total_bytes)}",
        ]
        self.summary_edit.setPlainText("\n".join(lines))
        self.status_label.setText("Estimate ready")
        self._append_log("Dataset size estimate updated.")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About ShapeDomainLab",
            (
                f"ShapeDomainLab\nVersion: {__version__}\n\n"
                "Generator of synthetic simple geometric shape datasets for "
                "computer vision experiments."
            ),
        )

    def _start_generation(self) -> None:
        try:
            config = self._build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Parameters", str(exc))
            return

        output_dir = resolve_output_dir(config.output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            QMessageBox.warning(self, "Invalid Output", f"Output path is not a directory:\n{output_dir}")
            return
        if output_dir.exists() and any(output_dir.iterdir()):
            try:
                existing_dataset = validate_generated_dataset_dir(output_dir)
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Output Directory Not Empty",
                    f"The output directory is not empty and is not a generated dataset.\n"
                    f"No files were removed:\n{output_dir}",
                )
                return
            reply = QMessageBox.question(
                self,
                "Replace Existing Dataset",
                f"Clear the existing dataset and generate a new one?\n\n{existing_dataset}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.status_label.setText("Generation cancelled")
                return
            try:
                clear_dataset_dir(existing_dataset)
            except (OSError, ValueError) as exc:
                QMessageBox.critical(self, "Clear Failed", str(exc))
                return

        self._set_running(True)
        self.progress_bar.setValue(0)
        self.log_edit.clear()
        self._append_log("Starting generation...")
        self._append_log(f"Output: {config.output_dir}")

        self._worker = GenerationWorker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _build_config(self) -> DatasetConfig:
        classes = tuple(name for name, checkbox in self.class_checks.items() if checkbox.isChecked())
        if not classes:
            raise ValueError("Select at least one shape class.")
        class_count = len(classes)

        background_range = self._range(
            self.background_min_spin.value(),
            self.background_max_spin.value(),
            "background",
        )
        foreground_range = self._range(
            self.foreground_min_spin.value(),
            self.foreground_max_spin.value(),
            "foreground",
        )
        line_width_range = self._range(
            self.line_min_spin.value(),
            self.line_max_spin.value(),
            "line width",
        )
        rotation_range = self._range(
            self.rotation_min_spin.value(),
            self.rotation_max_spin.value(),
            "rotation",
            allow_equal=True,
        )
        scale_range = self._range(
            self.scale_min_spin.value(),
            self.scale_max_spin.value(),
            "scale",
            allow_equal=True,
        )
        center_offset_range = self._range(
            self.center_offset_min_spin.value(),
            self.center_offset_max_spin.value(),
            "center offset",
            allow_equal=True,
        )

        file_format = self.format_combo.currentText()

        render_config = ShapeRenderConfig(
            color_mode=self.color_mode_combo.currentText(),
            background_range=background_range,
            foreground_range=foreground_range,
            background_color=self.background_color,
            foreground_color=self.foreground_color,
            line_width_range=line_width_range,
            fill_mode=self.fill_mode_combo.currentText(),
            rotation_range=rotation_range,
            scale_range=scale_range,
            center_offset_range=center_offset_range,
            edge_clipping=self.edge_clipping_check.isChecked(),
            min_visible_ratio=self.min_visible_ratio_spin.value(),
            noise_level=self.noise_level_spin.value() if self.noise_check.isChecked() else 0,
            margin=self.margin_spin.value(),
        )
        train_total = self.train_count_spin.value()
        val_total = self.val_count_spin.value()
        test_total = self.test_count_spin.value()
        if train_total % class_count != 0 or val_total % class_count != 0 or test_total % class_count != 0:
            raise ValueError(
                "Each subset count must be divisible by the number of selected classes so the generator can keep classes balanced."
            )
        return DatasetConfig(
            output_dir=Path(self.output_edit.text().strip()),
            image_size=int(self.image_size_combo.currentText()),
            classes=classes,
            train_count=train_total // class_count,
            val_count=val_total // class_count,
            test_count=test_total // class_count,
            subset_totals={"train": train_total, "val": val_total, "test": test_total},
            file_format=file_format,
            save_masks=self.save_masks_check.isChecked(),
            seed=self.seed_spin.value(),
            render=render_config,
        )

    def _apply_config(self, config: DatasetConfig) -> None:
        self.output_edit.setText(str(config.output_dir))
        self.image_size_combo.setCurrentText(str(config.image_size))
        class_count = max(1, len(config.classes))
        default_totals = {
            "train": config.train_count * class_count,
            "val": config.val_count * class_count,
            "test": config.test_count * class_count,
        }
        subset_totals = config.subset_totals or default_totals
        train_total = subset_totals.get("train", default_totals["train"])
        val_total = subset_totals.get("val", default_totals["val"])
        test_total = subset_totals.get("test", default_totals["test"])
        self._set_split_mode(automatic=False)
        self.train_count_spin.setValue(train_total)
        self.val_count_spin.setValue(val_total)
        self.test_count_spin.setValue(test_total)
        self.dataset_count_spin.setValue(train_total + val_total + test_total)
        self.seed_spin.setValue(config.seed)
        self.format_combo.setCurrentText(config.file_format.lower().lstrip("."))
        self.save_masks_check.setChecked(config.save_masks)

        for class_name, checkbox in self.class_checks.items():
            checkbox.setChecked(class_name in config.classes)

        render = config.render
        self.color_mode_combo.setCurrentText(render.color_mode)
        self.background_min_spin.setValue(render.background_range[0])
        self.background_max_spin.setValue(render.background_range[1])
        self.foreground_min_spin.setValue(render.foreground_range[0])
        self.foreground_max_spin.setValue(render.foreground_range[1])
        self.background_color = render.background_color
        self.foreground_color = render.foreground_color
        self._update_color_button(self.background_color_button, "Background color", self.background_color)
        self._update_color_button(self.foreground_color_button, "Shape color", self.foreground_color)
        self.line_min_spin.setValue(render.line_width_range[0])
        self.line_max_spin.setValue(render.line_width_range[1])
        self.fill_mode_combo.setCurrentText(render.fill_mode)
        self.rotation_min_spin.setValue(render.rotation_range[0])
        self.rotation_max_spin.setValue(render.rotation_range[1])
        self.scale_min_spin.setValue(render.scale_range[0])
        self.scale_max_spin.setValue(render.scale_range[1])
        self.center_offset_min_spin.setValue(render.center_offset_range[0])
        self.center_offset_max_spin.setValue(render.center_offset_range[1])
        self.edge_clipping_check.setChecked(render.edge_clipping)
        self.min_visible_ratio_spin.setValue(render.min_visible_ratio)
        self.min_visible_ratio_spin.setEnabled(render.edge_clipping)
        self.noise_check.setChecked(render.noise_level > 0)
        self.noise_level_spin.setValue(render.noise_level)
        self.margin_spin.setValue(render.margin)
        self._set_color_controls_enabled(render.color_mode)

    def _range(self, minimum, maximum, label: str, allow_equal: bool = False):
        invalid = minimum > maximum if allow_equal else minimum >= maximum
        if invalid:
            raise ValueError(f"The {label} minimum must be smaller than the maximum.")
        return (minimum, maximum)

    def _cancel_generation(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._append_log("Cancel requested...")
            self.cancel_button.setEnabled(False)

    def _clear_dataset(self) -> None:
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "Invalid Output", "Output directory cannot be empty.")
            return

        try:
            output_dir = validate_generated_dataset_dir(resolve_output_dir(Path(output_text)))
        except ValueError as exc:
            QMessageBox.warning(self, "Not a Dataset", str(exc))
            self._append_log(f"Clear skipped: {exc}")
            return

        reply = QMessageBox.question(
            self,
            "Clear Dataset",
            f"Delete all files and folders inside this dataset directory?\n\n{output_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            cleared_dir = clear_dataset_dir(output_dir)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Clear Failed", str(exc))
            self._append_log(f"Clear failed: {exc}")
            return

        self._last_output_dir = cleared_dir
        self.progress_bar.setValue(0)
        self.status_label.setText("Dataset cleared")
        self._append_log(f"Dataset cleared: {cleared_dir}")
        self._clear_preview()

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.status_label.setText(f"{done}/{total}")
        self._append_log(message)

    def _on_completed(self, output_dir: Path, report) -> None:
        self._last_output_dir = output_dir
        self._set_running(False)
        self.status_label.setText("Completed")
        self._append_log(f"Dataset generated: {output_dir}")
        if report.passed:
            self._append_log("Quality check: passed")
        else:
            self._append_log("Quality check: failed")
            for error in report.errors:
                self._append_log(f"- {error}")
        for warning in report.warnings:
            self._append_log(f"Quality warning: {warning}")
        self._append_log("Quality reports: quality_report.json, quality_report.md")
        self._load_preview(output_dir)

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self.status_label.setText("Failed")
        self._append_log(f"Error: {message}")
        QMessageBox.critical(self, "Generation Failed", message)

    def _on_cancelled(self) -> None:
        self._set_running(False)
        self.status_label.setText("Cancelled")
        self._append_log("Generation cancelled.")

    def _set_running(self, running: bool) -> None:
        self.generate_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.browse_button.setEnabled(not running)
        self.clear_button.setEnabled(not running)
        self.bottom_preview_button.setEnabled(not running)
        self.load_config_button.setEnabled(not running)
        self.save_config_button.setEnabled(not running)
        self.summary_button.setEnabled(not running)
        self.estimate_button.setEnabled(not running)
        self.generate_action.setEnabled(not running)
        self.cancel_action.setEnabled(running)
        self.clear_action.setEnabled(not running)
        self.refresh_preview_action.setEnabled(not running)
        self.load_config_action.setEnabled(not running)
        self.save_config_action.setEnabled(not running)
        self.summary_action.setEnabled(not running)
        self.estimate_action.setEnabled(not running)

    def _append_log(self, message: str) -> None:
        self.log_edit.append(message)

    def _clear_preview(self) -> None:
        for label in self.preview_labels:
            label.setText("No image")
            label.setPixmap(QPixmap())
            self._set_tooltip(label, "")

    def _refresh_preview(self) -> None:
        try:
            config = self._build_config()
            preview_images = generate_preview_images(config, len(self.preview_labels))
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Parameters", str(exc))
            return

        self._clear_preview()
        for index, preview_image in enumerate(preview_images):
            pixmap = self._pixmap_from_image(preview_image.image)
            label = self.preview_labels[index]
            if pixmap.isNull():
                label.setText(preview_image.class_name)
                continue
            label.setPixmap(pixmap.scaled(112, 112, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._set_tooltip(label, f"Preview: {preview_image.class_name}")
        self.status_label.setText("Preview refreshed")
        self._append_log("Preview refreshed from current parameters.")

    def _pixmap_from_image(self, image) -> QPixmap:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), "PNG")
        return pixmap

    def _load_preview_if_dataset(self) -> None:
        output_text = self.output_edit.text().strip()
        if not output_text:
            self._clear_preview()
            return

        try:
            output_dir = validate_generated_dataset_dir(resolve_output_dir(Path(output_text)))
        except ValueError:
            self._clear_preview()
            return

        self._last_output_dir = output_dir
        self.status_label.setText("Dataset loaded")
        self._append_log(f"Dataset preview loaded: {output_dir}")
        self._load_preview(output_dir)

    def _load_preview(self, output_dir: Path) -> None:
        images = self._preview_image_paths(output_dir, len(self.preview_labels))
        self._clear_preview()

        for label, image_path in zip(self.preview_labels, images):
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                label.setText(image_path.name)
                continue
            label.setPixmap(pixmap.scaled(112, 112, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._set_tooltip(label, str(image_path))

    def _preview_image_paths(self, output_dir: Path, limit: int) -> list[Path]:
        class_dirs = sorted(path for path in (output_dir / "train").iterdir() if path.is_dir())
        class_images = [
            sorted(path for path in class_dir.glob("*.*") if path.is_file())
            for class_dir in class_dirs
        ]

        images: list[Path] = []
        for paths in class_images:
            if paths:
                images.append(paths[0])
                if len(images) >= limit:
                    return images

        index = 1
        while len(images) < limit:
            added = False
            for paths in class_images:
                if index < len(paths):
                    images.append(paths[index])
                    added = True
                    if len(images) >= limit:
                        return images
            if not added:
                break
            index += 1
        return images

    def _format_summary(self, summary: DatasetSummary) -> str:
        lines = [
            f"Dataset: {summary.output_dir}",
            f"Exists: {'yes' if summary.exists else 'no'}",
            f"Total images: {summary.total_images}",
        ]
        for subset, class_counts in summary.subsets.items():
            counts = ", ".join(f"{name}={count}" for name, count in class_counts.items())
            lines.append(f"{subset}: {counts}" if counts else f"{subset}: empty")
        if summary.image_formats:
            formats = ", ".join(f"{name}={count}" for name, count in summary.image_formats.items())
            lines.append(f"Formats: {formats}")
        if summary.service_files:
            missing = [name for name, present in summary.service_files.items() if not present]
            lines.append(f"Missing service files: {', '.join(missing) if missing else 'none'}")
        return "\n".join(lines)

    def _format_bytes(self, value: int) -> str:
        size = float(value)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
            size /= 1024
        return f"{value} B"


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
