import sys
import cv2
import keyboard
import json
import os
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QImage, QPixmap, QPainterPath
from PyQt5.QtCore import Qt, QPoint, QTimer, QRectF
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QSlider,
                             QVBoxLayout, QHBoxLayout, QSizePolicy, QPushButton,
                             QColorDialog, QFrame, QComboBox)
import PyQt5.QtCore
import math
from config import load_config, save_config, get_default_config
from keybinds import KeybindsWindow
PyQt5.QtCore.QCoreApplication.setAttribute(PyQt5.QtCore.Qt.AA_EnableHighDpiScaling)
PyQt5.QtCore.QCoreApplication.setAttribute(PyQt5.QtCore.Qt.AA_UseHighDpiPixmaps)
import zhmiscellany
from logic import Logic
zhmiscellany.misc.die_on_key()

global primary_color, secondary_color, tertiary_color

# front end yapper, run this file

class Config:
    def __init__(self, initial_data=None):
        # Load defaults and override with provided data
        self.data = get_default_config()
        if initial_data:
            # Use the robust load_config result which already merges defaults
            self.data.update(initial_data)
        
        # Dynamic properties for convenience, mapping to self.data
        self.secondary_color = None
        self.tertiary_color = None
        self._update_colors()
    
    def __getattr__(self, name):
        """Allows accessing config values directly like cfg.SMOOTH_ALPHA."""
        if name in self.data:
            return self.data[name]
        # Allow access to logic/PyQt attributes or raise error if truly missing
        # FIX: The previous line 'return super().__getattribute__(self)' caused a TypeError.
        # When __getattr__ is called and the attribute is not found in custom logic,
        # the standard behavior is to raise an AttributeError.
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Allows setting config values directly and updates the data dictionary, saving on change."""
        if name in ["data", "secondary_color", "tertiary_color"]:
            # Allow setting internal attributes directly
            super().__setattr__(name, value)
        elif name in self.data:
            self.data[name] = value
            if name == "primary_color":
                self._update_colors()
            self.save_config()  # Save on any change to a persistent value
        else:
            # For other non-config attributes (like in logic.py or internal PyQt vars)
            super().__setattr__(name, value)
    
    def _update_colors(self):
        """Calculates secondary and tertiary colors based on the primary color."""
        self.secondary_color = darker(self.data["primary_color"], 0.7)
        self.tertiary_color = darker(self.data["primary_color"], 0.2)
    
    def save_config(self):
        """Saves the current configuration to file."""
        # Ensure we only save the primary persistent data
        save_config(self.data)


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def darker(hexcol, factor=0.7):
    r, g, b = hex_to_rgb(hexcol)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return rgb_to_hex((r, g, b))


class Overlay(QWidget):
    def __init__(self, screen_size):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, screen_size.width(), screen_size.height())
        self.pointer_pos = None
    
    def set_pointer_pos(self, pos):
        if pos is None:
            self.pointer_pos = None
        else:
            self.pointer_pos = QPoint(int(pos[0]), int(pos[1]))
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.pointer_pos:
            painter.setPen(QPen(Qt.red, 4))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self.pointer_pos, 20, 20)


class RoundedQLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = 12
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(rect, self.radius, self.radius)
        painter.setClipPath(path)
        pixmap = self.pixmap()
        if pixmap:
            # This scaling is handled by setScaledContents(True) on the QLabel itself.
            # The pixmap is automatically scaled to fit the label's geometry.
            scaled = pixmap.scaled(int(rect.width()), int(rect.height()),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (rect.width() - scaled.width()) / 2
            y = (rect.height() - scaled.height()) / 2
            painter.drawPixmap(int(x), int(y), scaled)
        else:
            super().paintEvent(event)


class AngleOffsetWidget(QWidget):
    def __init__(self, cfg, size=160, max_amount=1.0, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.circle_size = size
        self.setFixedSize(size, size)
        self.max_amount = float(max_amount)
        # Use cfg attributes which dynamically access the data dict
        self._angle = float(cfg.click_offset_deg % 360)
        self._amount = float(min(cfg.click_offset_amount, self.max_amount))
        self.dragging = False
        self.padding = 10
        self.snap_threshold_px = 10
        self.k_exp = 3.0
        self.setMouseTracking(True)
        self._deg_label = None
        self._strength_label = None
    
    def set_label_refs(self, deg_label, strength_label):
        self._deg_label = deg_label
        self._strength_label = strength_label
        self._update_labels()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width();
        h = self.height()
        cx = w / 2.0;
        cy = h / 2.0
        radius = min(w, h) / 2.0 - self.padding
        
        # Access colors via dynamic attributes
        border_col = QColor(darker(self.cfg.primary_color, 0.6))
        painter.setPen(QPen(border_col, 2))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(radius), int(radius))
        
        tick_col = QColor(darker(self.cfg.primary_color, 0.45))
        tick_pen = QPen(tick_col, 5)
        tick_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(tick_pen)
        inner_frac = 0.5
        outer_frac = 0.92
        for ang in (0, 90, 180, 270):
            rad = math.radians(ang)
            sx = cx + radius * inner_frac * math.cos(rad)
            sy = cy - radius * inner_frac * math.sin(rad)
            ex = cx + radius * outer_frac * math.cos(rad)
            ey = cy - radius * outer_frac * math.sin(rad)
            painter.drawLine(int(sx), int(sy), int(ex), int(ey))
        
        if self._amount <= 0.0:
            painter.setPen(Qt.NoPen)
            handle_col = QColor(self.cfg.secondary_color)
            painter.setBrush(QBrush(handle_col))
            painter.drawEllipse(QPoint(int(cx), int(cy)), 6, 6)
        else:
            denom = math.expm1(self.k_exp)
            ratio = max(0.0, min(1.0, float(self._amount) / max(self.max_amount, 1e-9)))
            vis_norm = math.log(ratio * denom + 1.0) / self.k_exp if denom > 0 else ratio
            vis_dist = vis_norm * radius
            rad = math.radians(self._angle)
            ex = cx + vis_dist * math.cos(rad)
            ey = cy - vis_dist * math.sin(rad)
            
            line_col = QColor(self.cfg.secondary_color)
            line_pen = QPen(line_col, 4)
            line_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(line_pen)
            start_frac = 0.08
            sx = cx + radius * start_frac * math.cos(rad)
            sy = cy - radius * start_frac * math.sin(rad)
            painter.drawLine(int(sx), int(sy), int(ex), int(ey))
            
            painter.setPen(QPen(QColor(darker(self.cfg.primary_color, 0.4)), 1))
            painter.setBrush(QBrush(line_col))
            painter.drawEllipse(QPoint(int(ex), int(ey)), 7, 7)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self._update_from_pos(event.pos())
    
    def mouseMoveEvent(self, event):
        if self.dragging:
            self._update_from_pos(event.pos())
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
    
    def _update_from_pos(self, qpoint):
        px = float(qpoint.x());
        py = float(qpoint.y())
        cx = self.width() / 2.0;
        cy = self.height() / 2.0
        dx = px - cx;
        dy = cy - py
        raw_dist = math.hypot(dx, dy)
        radius = min(self.width(), self.height()) / 2.0 - self.padding
        
        if raw_dist <= self.snap_threshold_px:
            # snap: set strength to 0 AND set degree to 0.0 as requested
            self._amount = 0.0
            self._angle = 0.0
            # Use setter which calls save_config
            self.cfg.click_offset_amount = 0.0
            self.cfg.click_offset_deg = 0.0
            self._update_labels()
            self.update()
            return
        
        angle = (math.degrees(math.atan2(dy, dx))) % 360.0
        norm = max(0.0, min(1.0, raw_dist / max(radius, 1e-9)))
        denom = math.expm1(self.k_exp)
        amount = float(self.max_amount * (math.expm1(norm * self.k_exp) / max(denom, 1e-9)))
        amount = max(0.0, min(self.max_amount, amount))
        
        self._angle = float(angle);
        self._amount = float(amount)
        # Use setters which call save_config
        self.cfg.click_offset_deg = float(self._angle)
        self.cfg.click_offset_amount = float(self._amount)
        self._update_labels()
        self.update()
    
    def set_from_cfg(self):
        # Read from cfg attributes
        self._angle = float(self.cfg.click_offset_deg % 360)
        self._amount = float(min(self.cfg.click_offset_amount, self.max_amount))
        self._update_labels()
        self.update()
    
    def _update_labels(self):
        if self._deg_label is not None:
            self._deg_label.setText(f"Offset Degree: {self._angle:.1f}°")
        if self._strength_label is not None:
            self._strength_label.setText(f"Offset Strength: {self._amount:.2f}")


class CustomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setFixedHeight(30)
        self.setStyleSheet("background-color: #2c3e50;")
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.title = QLabel("Hand Mouse")
        self.title.setStyleSheet("font-size: 16px; color: white; font-weight: bold; background: transparent;")
        self.title.setAttribute(Qt.WA_TranslucentBackground)
        self.layout.addWidget(self.title)
        self.layout.addStretch(1)
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setStyleSheet("QPushButton {border: none; background-color: #2c3e50; color: white;} QPushButton:hover {background-color: #e74c3c;}")
        self.close_btn.clicked.connect(self.parent.close)
        self.layout.addWidget(self.close_btn)
        self.start_pos = None
        self.end_pos = None
        self.is_dragging = False
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.start_pos = event.globalPos()
            self.end_pos = self.parent.pos()
    
    def mouseMoveEvent(self, event):
        if self.is_dragging:
            delta = event.globalPos() - self.start_pos
            self.parent.move(self.end_pos + delta)
    
    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.start_pos = None
        self.end_pos = None


class MainWidget(QWidget):
    def __init__(self, cam_label, options_frame, instruction_label):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setObjectName("mainFrame")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Add the instruction label at the very top, just below the title bar
        self.instruction_label = instruction_label
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setStyleSheet("font-size: 16px; color: #ecf0f1; background: #34495e; border-radius: 4px; padding: 5px;")
        
        self.title_bar = CustomTitleBar(self)
        self.layout.addWidget(self.title_bar)
        self.layout.addWidget(self.instruction_label)  # NEW: Add instruction label
        
        self.content_layout = QHBoxLayout()
        self.content_layout.addWidget(cam_label, stretch=1)  # Video preview stretches
        self.content_layout.addStretch()  # This will push the options frame to the right
        self.content_layout.addWidget(options_frame, stretch=0)
        self.layout.addLayout(self.content_layout)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 12, 12)
        bg = self.palette().color(self.backgroundRole())
        painter.fillPath(path, QBrush(bg))


def mk_slider(minv, maxv, start, step=1):
    s = QSlider(Qt.Horizontal)
    s.setRange(minv, maxv)
    s.setSingleStep(step)
    s.setValue(start)
    return s


def apply_styles(win, cfg, options_frame, widgets):
    # Colors read dynamically from cfg object
    global primary_color, secondary_color, tertiary_color
    primary, primary_color = cfg.primary_color, cfg.primary_color
    secondary, secondary_color = cfg.secondary_color, cfg.secondary_color
    tertiary, tertiary_color = cfg.tertiary_color, cfg.tertiary_color
    
    win.setStyleSheet(f"""
        #mainFrame {{
            background: {secondary};
            border-radius: 12px;
            border: 2px solid {secondary};
        }}
        #optionsFrame {{
            background: {primary};
            border-radius: 8px;
            padding: 15px;
        }}
        QLabel {{ color: #fff; }}
        QLabel#optionsHeader {{
            font-weight: 700;
            font-size: 20px;
            color: #fff;
            background: transparent;
        }}
        #camDisplay {{
            border-radius: 8px;
            border: 2px solid {secondary};
            background: #000;
        }}
    """)
    for w in widgets:
        if isinstance(w, QLabel):
            w.setStyleSheet("color: #fff; background: transparent;")
        elif isinstance(w, QPushButton):
            w.setStyleSheet(f"""
                QPushButton{{
                    background:{secondary};
                    color:#ffffff;
                    border:none;
                    border-radius:6px;
                    padding:8px 12px;
                    font-weight: bold;
                    font-size: 15px;
                }}
                QPushButton:hover{{
                    background:{darker(secondary, 0.5)};
                }}
            """)
        elif isinstance(w, QSlider):
            w.setStyleSheet(f"""
                QSlider::groove:horizontal{{
                    height: 8px;
                    background: {tertiary};
                    border-radius: 4px;
                }}
                QSlider::handle:horizontal{{
                    width: 16px;
                    margin: -4px 0;
                    border-radius: 8px;
                    background: {secondary};
                }}
            """)
        elif isinstance(w, QFrame):
            w.setStyleSheet(f"""
                QFrame {{
                    background: {primary};
                    border-radius: 8px;
                    padding: 15px;
                }}
                QLabel {{ color: #fff; background: transparent; }}
                QLabel#optionsHeader {{
                    font-weight: 700;
                    font-size: 16px;
                    color: #fff;
                    background: transparent;
                }}
            """)


def detect_cameras(max_probe=6):
    cams = []
    for i in range(0, max_probe + 1):
        cap = cv2.VideoCapture(i)
        ok, _ = cap.read()
        if ok:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cams.append((i, width, height))
        try:
            cap.release()
        except Exception:
            pass
    return cams


def main():
    # Load config data and initialize Config object
    initial_data = load_config()
    cfg = Config(initial_data)
    
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    W, H = screen.geometry().width(), screen.geometry().height()
    
    # Store the KeybindsWindow instance on the app object to prevent garbage collection
    app.keybinds_win = None
    
    logic = Logic(cfg)
    logic.capture_key = cfg.capture_key  # Update logic with loaded keybind
    
    overlay = Overlay(screen.geometry())
    overlay.show()
    
    # Initialize cam_label here, before it's referenced in nested functions
    cam_label = RoundedQLabel()
    cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    cam_label.setMinimumSize(640, 360)
    cam_label.setScaledContents(True)  # Crucial for scaling
    
    # NEW: Initialize the instruction label
    instruction_label = QLabel("1 sec")
    instruction_label.setObjectName("instructionLabel")
    
    options_frame = QFrame()
    options_frame.setObjectName("optionsFrame")
    options_frame.setFixedWidth(320)  # fixed width
    options_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    vbox = QVBoxLayout(options_frame)
    header = QLabel("Options")
    header.setObjectName("optionsHeader")
    header.setAlignment(Qt.AlignCenter)
    
    smooth_label = QLabel(f"Smooth Alpha: {cfg.SMOOTH_ALPHA:.2f}")
    smooth_s = mk_slider(0, 100, int(cfg.SMOOTH_ALPHA * 100))
    
    def upd_smooth(v):
        # Use cfg setter to automatically save
        cfg.SMOOTH_ALPHA = v / 100.0
        smooth_label.setText(f"Smooth Alpha: {cfg.SMOOTH_ALPHA:.2f}")
    
    smooth_s.valueChanged.connect(upd_smooth)
    
    fast_label = QLabel(f"Fast Movement Alpha: {cfg.FAST_MOVEMENT_ALPHA:.2f}")
    fast_s = mk_slider(0, 100, int(cfg.FAST_MOVEMENT_ALPHA * 100))
    
    def upd_fast(v):
        # Use cfg setter to automatically save
        cfg.FAST_MOVEMENT_ALPHA = v / 100.0
        fast_label.setText(f"Fast Movement Alpha: {cfg.FAST_MOVEMENT_ALPHA:.2f}")
    
    fast_s.valueChanged.connect(upd_fast)
    
    thresh_label = QLabel(f"Fast Movement Threshold: {cfg.FAST_MOVEMENT_THRESHOLD * 100:.1f}%")
    thresh_s = mk_slider(0, 100, int(cfg.FAST_MOVEMENT_THRESHOLD * 100))
    
    def upd_thresh(v):
        # Use cfg setter to automatically save
        cfg.FAST_MOVEMENT_THRESHOLD = v / 100.0
        thresh_label.setText(f"Fast Movement Threshold: {cfg.FAST_MOVEMENT_THRESHOLD * 100:.1f}%")
    
    thresh_s.valueChanged.connect(upd_thresh)
    
    fov_label = QLabel(f"Fov Factor: {cfg.fov_factor:.2f}")
    fov_s = mk_slider(0, 100, int(cfg.fov_factor * 100))
    
    def upd_fov(v):
        # Use cfg setter to automatically save
        cfg.fov_factor = v / 100.0
        fov_label.setText(f"Fov Factor: {cfg.fov_factor:.2f}")
    
    fov_s.valueChanged.connect(upd_fov)
    
    raw_mode_threshold_label = QLabel(f"Mode Switch Threshold: {cfg.fov_factor:.2f}")
    raw_mode_threshold_s = mk_slider(0, 100, int(cfg.fov_factor * 100))
    
    def upd_raw_mode_threshold(v):
        # Use cfg setter to automatically save
        cfg.raw_mode_threshold_factor = v / 100.0
        raw_mode_threshold_label.setText(f"Mode Switch Threshold: {cfg.raw_mode_threshold_factor:.2f}")
    
    raw_mode_threshold_s.valueChanged.connect(upd_raw_mode_threshold)
    
    raw_mode_switch_time_label = QLabel(f"Mode Switch Time: {cfg.raw_mode_switch_time:.2f}")
    raw_mode_switch_time_s = mk_slider(0, 400, int(cfg.raw_mode_switch_time * 100))
    
    def upd_raw_mode_switch_time(v):
        cfg.raw_mode_switch_time = v / 100.0
        raw_mode_switch_time_label.setText(f"Mode Switch Time: {cfg.raw_mode_switch_time:.2f}")
    
    raw_mode_switch_time_s.valueChanged.connect(upd_raw_mode_switch_time)
    
    raw_x_label = QLabel(f"Raw Mode Sensitivity X: {cfg.raw_mode_sensitivity_x:.4f}")
    raw_x_s = mk_slider(0, 1000, int(cfg.raw_mode_sensitivity_x * 100000))
    
    def upd_raw_x(v):
        # Use cfg setter to automatically save
        cfg.raw_mode_sensitivity_x = v / 100000.0
        raw_x_label.setText(f"Raw Mode Sensitivity X: {cfg.raw_mode_sensitivity_x:.4f}")
    
    raw_x_s.valueChanged.connect(upd_raw_x)
    
    raw_y_label = QLabel(f"Raw Mode Sensitivity Y: {cfg.raw_mode_sensitivity_y:.4f}")
    raw_y_s = mk_slider(0, 1000, int(cfg.raw_mode_sensitivity_y * 100000))
    
    def upd_raw_y(v):
        # Use cfg setter to automatically save
        cfg.raw_mode_sensitivity_y = v / 100000.0
        raw_y_label.setText(f"Raw Mode Sensitivity Y: {cfg.raw_mode_sensitivity_y:.4f}")
    
    raw_y_s.valueChanged.connect(upd_raw_y)
    
    raw_dead_label = QLabel(f"Raw Mode Deadzone: {cfg.raw_mode_deadzone:.2f}")
    raw_dead_s = mk_slider(0, 100, int(cfg.raw_mode_deadzone * 100))
    
    def upd_dead(v):
        cfg.raw_mode_deadzone = v / 100.0
        raw_dead_label.setText(f"Raw Mode Deadzone: {cfg.raw_mode_deadzone:.2f}")
    
    raw_dead_s.valueChanged.connect(upd_dead)
    
    calib_layout = QHBoxLayout()
    capture_btn = QPushButton(f"Calibrated {len(logic.image_fingertip_points)}/9")
    keybind_btn = QPushButton(f"'{cfg.capture_key.upper()}'")
    calib_layout.addWidget(capture_btn)
    calib_layout.addWidget(keybind_btn)
    
    def update_capture_text():
        capture_btn.setText(f"Calibrated {len(logic.image_fingertip_points)}/9")
    
    def on_capture_clicked():
        logic.capture_requested = True
    
    capture_btn.clicked.connect(on_capture_clicked)
    
    def on_global_key(event):
        if getattr(logic, "suppress_capture", False):
            return
        if hasattr(event, "name") and event.name == cfg.capture_key:
            logic.capture_requested = True
    
    keyboard.on_press(on_global_key)
    
    temp_hook = [None]
    
    def on_set_keybind_clicked():
        keybind_btn.setText("Listening...")
        keybind_btn.setEnabled(False)
        logic.suppress_capture = True
        
        if temp_hook[0] is not None:
            keyboard.unhook(temp_hook[0])
            temp_hook[0] = None
        
        def _once(event):
            name = getattr(event, "name", None)
            if not name:
                return
            
            # Update config and logic with the new keybind (setter calls save_config)
            cfg.capture_key = name
            logic.capture_key = name
            
            keybind_btn.setText(f"'{cfg.capture_key.upper()}'")
            keybind_btn.setEnabled(True)
            logic.suppress_capture = False
            if temp_hook[0] is not None:
                keyboard.unhook(temp_hook[0])
                temp_hook[0] = None
        
        temp_hook[0] = keyboard.on_press(_once)
    
    keybind_btn.clicked.connect(on_set_keybind_clicked)
    
    color_btn = QPushButton("Select Primary Color")
    color_display = QLabel(cfg.primary_color)
    color_display.setAlignment(Qt.AlignCenter)
    
    def pick_color():
        c = QColorDialog.getColor(QColor(cfg.primary_color), main_win, "Select Primary Color")
        if c.isValid():
            # Setting cfg.primary_color triggers the setter, which calls _update_colors and save_config
            cfg.primary_color = c.name()
            
            # Re-apply styles after color change
            color_display.setText(cfg.primary_color)
            apply_styles(main_win, cfg, options_frame, style_widgets)
    
    color_btn.clicked.connect(pick_color)
    
    # --- Keybinds Window Integration ---
    keybind_edit_btn = QPushButton("Raw Mode Keybinds/Thresholds")
    
    def on_edit_keybinds_clicked():
        # Pass the shared Config instance to the KeybindsWindow
        # Create the window and store the reference on the app object
        global primary_color, secondary_color, tertiary_color
        app.keybinds_win = KeybindsWindow(config=cfg, primary_color=primary_color, secondary_color=secondary_color, tertiary_color=tertiary_color)
        app.keybinds_win.show()
    
    keybind_edit_btn.clicked.connect(on_edit_keybinds_clicked)
    # --- End Keybinds Window Integration ---
    
    vbox.addWidget(header)
    vbox.addStretch(1)
    vbox.addLayout(calib_layout)
    vbox.addStretch(1)
    vbox.addWidget(smooth_label)
    vbox.addWidget(smooth_s)
    vbox.addWidget(fast_label)
    vbox.addWidget(fast_s)
    vbox.addWidget(thresh_label)
    vbox.addWidget(thresh_s)
    vbox.addWidget(fov_label)
    vbox.addWidget(fov_s)
    vbox.addWidget(raw_mode_threshold_label)
    vbox.addWidget(raw_mode_threshold_s)
    vbox.addWidget(raw_mode_switch_time_label)
    vbox.addWidget(raw_mode_switch_time_s)
    vbox.addWidget(raw_x_label)
    vbox.addWidget(raw_x_s)
    vbox.addWidget(raw_y_label)
    vbox.addWidget(raw_y_s)
    vbox.addWidget(raw_dead_label)
    vbox.addWidget(raw_dead_s)
    
    # ... (Camera/Angle Widgets)
    cams = detect_cameras(6)
    cam_combo = QComboBox()
    for idx, w, h in cams:
        display = f"Camera {idx} ({w}x{h})" if w and h else f"Camera {idx}"
        cam_combo.addItem(display, idx)
    
    if cams:
        cam_combo.setEnabled(True)
    else:
        cam_combo.addItem("No camera found", -1)
        cam_combo.setEnabled(False)
    
    # Define on_camera_changed after cam_label is fully initialized
    def on_camera_changed(i):
        data = cam_combo.itemData(i)
        if data is None or data == -1:
            return
        ok = logic.set_camera(int(data))
        # after switching, query actual resolution from camera
        nw, nh = logic.get_camera_resolution()
        if nw == 0 or nh == 0:
            nw, nh = 640, 480
        # Removed cam_label.setFixedSize(nw, nh) to allow expansion
    
    cam_combo.currentIndexChanged.connect(on_camera_changed)
    
    # apply current selection immediately (ensures feed appears)
    if cam_combo.count() > 0 and cam_combo.isEnabled():
        on_camera_changed(cam_combo.currentIndex())
    pinky_label = QLabel("Pinky Offset")
    pinky_label.setAlignment(Qt.AlignCenter)
    pinky_label.setStyleSheet("font-weight:700; font-size:16px; color:#fff; background:transparent;")
    
    deg_label = QLabel(f"Offset Degree: {cfg.click_offset_deg:.1f}°")
    deg_label.setAlignment(Qt.AlignCenter)
    deg_label.setStyleSheet("font-size:16px; color:#ddd; background:transparent;")
    
    strength_label = QLabel(f"Offset Strength: {cfg.click_offset_amount:.2f}")
    strength_label.setAlignment(Qt.AlignCenter)
    strength_label.setStyleSheet("font-size:16px; color:#ddd; background:transparent;")
    
    angle_widget = AngleOffsetWidget(cfg, size=160, max_amount=1.0)
    angle_widget.set_from_cfg()
    angle_widget.set_label_refs(deg_label, strength_label)
    
    wrapper = QWidget()
    wrapper_layout = QVBoxLayout(wrapper)
    wrapper_layout.setContentsMargins(0, 0, 0, 0)
    wrapper_layout.setSpacing(6)
    wrapper_layout.addWidget(pinky_label)
    wrapper_layout.addWidget(deg_label)
    wrapper_layout.addWidget(strength_label)
    hbox = QHBoxLayout()
    hbox.addStretch()
    hbox.addWidget(angle_widget)
    hbox.addStretch()
    wrapper_layout.addLayout(hbox)
    vbox.addWidget(wrapper)
    
    vbox.addWidget(cam_combo)
    
    # Placement for the new button
    vbox.addStretch(1)  # Decrease stretch above to make room
    vbox.addWidget(keybind_edit_btn)  # NEW BUTTON
    vbox.addStretch(24)  # Add back remaining stretch to push color controls down
    
    vbox.addWidget(color_btn)
    vbox.addWidget(color_display)
    
    # NEW: Pass the instruction_label to MainWidget
    main_win = MainWidget(cam_label, options_frame, instruction_label)
    main_win.resize(int(W * 0.45), int(H * 0.45))
    main_win.show()
    
    style_widgets = [
        header, color_btn, color_display, smooth_s, fast_s, thresh_s,
        fov_s, smooth_label, fast_label, thresh_label, fov_label, raw_x_label, raw_mode_threshold_s, raw_mode_switch_time_s, raw_x_s, raw_y_s,
        raw_mode_threshold_label, raw_mode_switch_time_label, raw_y_label, raw_dead_label, raw_dead_s,
        capture_btn, keybind_btn, cam_combo, keybind_edit_btn, instruction_label  # ADDED instruction_label HERE
    ]
    
    # Apply initial styles
    apply_styles(main_win, cfg, options_frame, style_widgets)
    
    # NEW: State variable to persist the last non-None instruction
    last_instruction = "1 sec"
    
    def tick():
        nonlocal last_instruction
        img, pointer, quad, instruction = logic.process_frame(W, H)
        if img is None:
            return
        overlay.set_pointer_pos(pointer)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        cam_label.setPixmap(pix)
        
        # NEW: Update the instruction label text with persistence logic
        if instruction is not None:
            last_instruction = instruction
        
        instruction_label.setText(last_instruction)
        
        update_capture_text()
    
    timer = QTimer()
    # FIX: Use the loaded config value for frame interval
    timer.setInterval(cfg.frame_interval_ms)
    timer.timeout.connect(tick)
    timer.start()
    
    exit_code = app.exec_()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()