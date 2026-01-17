import sys
import os
import keyboard
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QApplication, QSlider, QPushButton
)
from PyQt5.QtCore import Qt, QPoint, QRect, QCoreApplication, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QTransform, QColor, QFont, QFontMetrics

QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

try:
    from pynput import mouse as pynput_mouse
    
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False


class KeyAssignmentWindow(QWidget):
    def __init__(self, finger_key, current_keybind, config, primary_color, secondary_color, tertiary_color, parent=None):
        super().__init__(parent)
        self.finger_key = finger_key
        self.config = config
        self.main_window = parent
        self.primary_color, self.secondary_color, self.tertiary_color = primary_color, secondary_color, tertiary_color
        self.assigned_key, self._keyboard_hook, self._pynput_mouse_listener = "", None, None
        
        self.setWindowTitle(f"Assign Key for {finger_key}")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setFixedSize(380, 160);
        self.setWindowModality(Qt.ApplicationModal)
        
        self.layout = QVBoxLayout(self)
        self.info_label = QLabel(f"Assign key for: {self.finger_key}");
        self.info_label.setAlignment(Qt.AlignCenter)
        self.current_key_label = QLabel();
        self.current_key_label.setAlignment(Qt.AlignCenter)
        self.key_display_edit = QLineEdit(placeholderText="Press any key or click a mouse button...")
        self.key_display_edit.setReadOnly(True);
        self.key_display_edit.setAlignment(Qt.AlignCenter)
        
        self.layout.addWidget(self.info_label);
        self.layout.addWidget(self.current_key_label);
        self.layout.addWidget(self.key_display_edit)
        
        self.setStyleSheet(f"""
            QWidget {{ background-color: {self.primary_color}; color: white; border-radius: 8px; }}
            QLabel {{ color: white; padding: 5px; }}
            QLineEdit {{ background-color: {self.primary_color}; color: white;
                border: 2px solid {self.tertiary_color}; padding: 5px; border-radius: 4px; }}
        """)
        
        self._set_current_bind_display(current_keybind)
        if self.main_window: self.main_window.setEnabled(False)
        QTimer.singleShot(250, self.start_listening)
    
    def _set_current_bind_display(self, raw):
        text = self._humanize(raw);
        font = QFont();
        font.setBold(True)
        max_size, min_size, fm = 16, 8, QFontMetrics(font)
        for size in range(max_size, min_size - 1, -1):
            font.setPointSize(size);
            fm = QFontMetrics(font)
            if fm.horizontalAdvance(text) <= (self.width() - 40): break
        self.current_key_label.setFont(font)
        self.current_key_label.setText(f"Current Bind: <b>{text if text else '&nbsp;'}</b>")
    
    def _humanize(self, raw):
        if not raw or raw == "N/A": return ""
        return f"{raw.split('mouse:')[1].replace('_', ' ').upper()} CLICK" if isinstance(raw, str) and raw.startswith("mouse:") else str(raw).upper()
    
    def start_listening(self):
        if not self.isVisible(): return
        
        def kb_once(event):
            try:
                key_name = getattr(event, "name", None)
                if key_name and self.isActiveWindow() and key_name not in ['shift', 'ctrl', 'alt']:
                    self._assign_and_close(key_name)
            except Exception:
                pass
        
        try:
            self._keyboard_hook = keyboard.on_press(kb_once)
        except Exception:
            self._keyboard_hook = None
        
        if PYNPUT_AVAILABLE:
            def on_click(x, y, button, pressed):
                if pressed and self.isActiveWindow():
                    self._assign_and_close(f"mouse:{str(button).split('.')[-1]}", mouse=True)
            
            def on_scroll(x, y, dx, dy):
                if self.isActiveWindow():
                    if dy > 0:
                        # Mouse scroll up
                        self._assign_and_close("mouse:scroll_up", mouse=True)
                    elif dy < 0:
                        # Mouse scroll down
                        self._assign_and_close("mouse:scroll_down", mouse=True)
            
            QTimer.singleShot(150, lambda: self._start_pynput_listener(on_click, on_scroll))
    
    def _start_pynput_listener(self, click_callback, scroll_callback):
        if not PYNPUT_AVAILABLE: return
        try:
            if self._pynput_mouse_listener: self._pynput_mouse_listener.stop(); self._pynput_mouse_listener = None
            self._pynput_mouse_listener = pynput_mouse.Listener(on_click=click_callback, on_scroll=scroll_callback)
            self._pynput_mouse_listener.start()
        except Exception:
            self._pynput_mouse_listener = None
    
    def _assign_and_close(self, key_value, mouse: bool = False):
        store = key_value
        display = self._humanize(store)
        self.assigned_key = store
        self.key_display_edit.setText(display)
        
        try:
            mode_key, threshold_key = self.main_window.current_mode_key, self.main_window.current_threshold_key
            
            # --- FIX: Use correct config key structure ---
            # Keybinds are saved under the mode key (e.g., 'n_keybinds') using the finger_key as the dict key.
            self.config.data.setdefault(mode_key, {})[self.finger_key] = self.assigned_key
            # Thresholds are saved under the threshold key (e.g., 'n_thresholds') using the finger_key as the dict key.
            # Only set default if it's not already there. This ensures Normal Mode fingers (e.g., "Pinky")
            # save their data to the correct keys and don't overwrite the entire dict.
            if self.finger_key not in self.config.data.setdefault(threshold_key, {}):
                self.config.data[threshold_key][self.finger_key] = 0.5
            # --- END FIX ---
            
            self.config.save_config()
            if self.parent() and hasattr(self.parent(), 'update_keybind_displays'): self.parent().update_keybind_displays()
            
            print(f"=== {mode_key.upper()} and {threshold_key.upper()} After Assignment ===")
            current_keybinds, current_thresholds = self.config.data.get(mode_key, {}), self.config.data.get(threshold_key, {})
            for k, v in sorted(current_keybinds.items()):
                threshold = current_thresholds.get(k, "N/A")
                key_pretty = self._humanize(v) if v else "None"
                threshold_pretty = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else str(threshold)
                print(f"{k}: Key={key_pretty}, Threshold={threshold_pretty}")
            print("=============================================")
        except Exception as e:
            print(f"Error saving/printing config: {e}")
        
        self._cleanup_listeners();
        self.close()
    
    def _cleanup_listeners(self):
        if self._keyboard_hook:
            try:
                keyboard.unhook(self._keyboard_hook)
            except Exception:
                pass
            self._keyboard_hook = None
        if PYNPUT_AVAILABLE and self._pynput_mouse_listener:
            try:
                self._pynput_mouse_listener.stop()
            except Exception:
                pass
            self._pynput_mouse_listener = None
    
    def closeEvent(self, event):
        self._cleanup_listeners()
        if self.main_window: self.main_window.setEnabled(True)
        super().closeEvent(event)


class HandImageWidget(QLabel):
    FINGER_REGIONS_NORM = {
        "Pinky": QRect(790, 250, 200, 600), "Ring": QRect(640, 50, 230, 650), "Middle": QRect(430, 0, 220, 700),
        "Index": QRect(225, 50, 235, 650), "Thumb": QRect(0, 300, 300, 700),
    }
    NORMAL_MODE_EDITABLE_FINGERS = ["Thumb", "Middle", "Pinky"]
    
    def __init__(self, hand_side, config, primary_color, secondary_color, tertiary_color, parent=None):
        super().__init__(parent)
        self.hand_side = hand_side
        self.config = config
        self.primary_color, self.secondary_color, self.tertiary_color = primary_color, secondary_color, tertiary_color
        self.setScaledContents(True);
        self.setCursor(Qt.PointingHandCursor);
        self.setMouseTracking(True)
        self.setMinimumSize(250, 300);
        self.current_highlighted_finger = None;
        self.key_assignment_window = None
        self.sliders = {};
        self.main_window = parent
        
        image_path = os.path.join(os.path.dirname(__file__), r"assets\hand.png")
        self.base_pixmap = QPixmap(image_path)
        if self.base_pixmap.isNull(): self.base_pixmap = QPixmap(300, 400); self.base_pixmap.fill(QColor(primary_color))
        
        slider_style = f"""
            QSlider::groove:horizontal{{ height: 8px; background: {self.tertiary_color}; border-radius: 4px; }}
            QSlider::handle:horizontal{{ width: 16px; margin: -4px 0; border-radius: 8px; background: {self.secondary_color}; }}
        """
        self.setStyleSheet(f"""
            HandImageWidget {{ background-color: {secondary_color}; border: 2px solid {tertiary_color}; border-radius: 8px; }}
        """ + slider_style)
        
        self._initialize_sliders(slider_style)
        self.update_hand_display()
        self.update_keybind_displays()
    
    def _get_key_for_mode(self, finger_name):
        """Returns the config key: 'Pinky' for Normal Mode, 'Left Hand - Pinky' for Raw Mode."""
        if self.main_window.current_mode == self.main_window.MODE_NORMAL:
            return finger_name
        return f"{self.hand_side} Hand - {finger_name}"
    
    def _is_editable(self, finger_name):
        if self.main_window.current_mode == self.main_window.MODE_RAW: return True
        return self.hand_side == self.main_window.selected_hand and finger_name in self.NORMAL_MODE_EDITABLE_FINGERS
    
    def update_hand_display(self):
        pixmap = self.base_pixmap
        is_normal_right = self.main_window.current_mode == self.main_window.MODE_NORMAL and self.main_window.selected_hand == "Right"
        is_raw_right = self.main_window.current_mode == self.main_window.MODE_RAW and self.hand_side == "Right"
        if is_normal_right or is_raw_right:
            pixmap = pixmap.transformed(QTransform().scale(-1, 1))
        self.setPixmap(pixmap)
        self.resize_and_position_sliders()
    
    def _initialize_sliders(self, style):
        for finger_name in self.FINGER_REGIONS_NORM.keys():
            # Sliders are initialized with the Raw Mode full key for tracking, but will use the correct key
            # when updating config/display
            for side in ["Left", "Right"]:
                full_key_raw = f"{side} Hand - {finger_name}"
                slider = QSlider(Qt.Horizontal, self);
                slider.setRange(0, 100);
                slider.setStyleSheet(style)
                slider.setMinimumWidth(80);
                slider.setMaximumWidth(150)
                # Pass a lambda function that resolves the *current* key when updating the threshold
                slider.valueChanged.connect(lambda value, key_raw=full_key_raw: self._update_threshold(key_raw, value))
                self.sliders[full_key_raw] = {'slider': slider, 'finger_name': finger_name};
                slider.hide()
    
    def _update_threshold(self, full_key_raw, value):
        # Determine the actual config key to save to based on the current mode
        finger_name = self.sliders[full_key_raw]['finger_name']
        config_key = self._get_key_for_mode(finger_name)
        
        threshold_key = self.main_window.current_threshold_key
        self.config.data.setdefault(threshold_key, {})[config_key] = value / 100.0
        self.config.save_config();
        self.update()
    
    def update_keybind_displays(self):
        mode_key, threshold_key = self.main_window.current_mode_key, self.main_window.current_threshold_key
        current_keybinds = self.config.data.get(mode_key, {})
        current_thresholds = self.config.data.get(threshold_key, {})
        
        for side in ["Left", "Right"]:
            for full_key_raw, data in self.sliders.items():
                if not full_key_raw.startswith(f"{side} Hand -") or side != self.hand_side: continue
                
                slider, finger_name = data['slider'], data['finger_name']
                is_visible = (self.main_window.current_mode == self.main_window.MODE_RAW) or \
                             (self.main_window.current_mode == self.main_window.MODE_NORMAL and side == self.main_window.selected_hand)
                
                # Get the *actual* config key
                config_key = self._get_key_for_mode(finger_name)
                
                keybind_raw = current_keybinds.get(config_key)
                is_editable = self._is_editable(finger_name)
                
                if is_visible and (keybind_raw is not None or keybind_raw == ""):
                    current_threshold = current_thresholds.get(config_key, 0.5)
                    slider.setValue(int(current_threshold * 100))
                    slider.setVisible(is_editable)
                else:
                    slider.hide()
        
        self.resize_and_position_sliders();
        self.update()
    
    def resizeEvent(self, event):
        super().resizeEvent(event);
        self.resize_and_position_sliders()
    
    def resize_and_position_sliders(self):
        w_scale, h_scale = self.width() / 1000.0, self.height() / 1000.0
        is_flipped = (self.main_window.current_mode == self.main_window.MODE_RAW and self.hand_side == "Left") or \
                     (self.main_window.current_mode == self.main_window.MODE_NORMAL and self.main_window.selected_hand == "Left")
        
        for full_key_raw, data in self.sliders.items():
            slider = data['slider']
            if full_key_raw.split(" Hand -")[0] != self.hand_side or not slider.isVisible(): continue
            
            norm_rect = self.FINGER_REGIONS_NORM[data['finger_name']]
            x, y, w, h = int(norm_rect.x() * w_scale), int(norm_rect.y() * h_scale), \
                int(norm_rect.width() * w_scale), int(norm_rect.height() * h_scale)
            if is_flipped: x = self.width() - (x + w)
            
            slider_w, slider_h = min(w - 10, 100), 10
            slider_x, slider_y = x + (w - slider_w) // 2, y + int(h * 0.45)
            slider.setGeometry(slider_x, slider_y, slider_w, slider_h)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        w_scale, h_scale = self.width() / 1000.0, self.height() / 1000.0
        painter = QPainter(self);
        painter.setRenderHint(QPainter.Antialiasing)
        
        mode_key, threshold_key = self.main_window.current_mode_key, self.main_window.current_threshold_key
        current_keybinds, current_thresholds = self.config.data.get(mode_key, {}), self.config.data.get(threshold_key, {})
        is_flipped = (self.main_window.current_mode == self.main_window.MODE_RAW and self.hand_side == "Left") or \
                     (self.main_window.current_mode == self.main_window.MODE_NORMAL and self.main_window.selected_hand == "Left")
        
        for finger_name, norm_rect in self.FINGER_REGIONS_NORM.items():
            config_key = self._get_key_for_mode(finger_name)
            keybind_raw = current_keybinds.get(config_key)
            x, y, w, h = int(norm_rect.x() * w_scale), int(norm_rect.y() * h_scale), \
                int(norm_rect.width() * w_scale), int(norm_rect.height() * h_scale)
            if is_flipped: x = self.width() - (x + w)
            highlight_rect, is_editable = QRect(x, y, w, h), self._is_editable(finger_name)
            
            text_visible = (self.main_window.current_mode == self.main_window.MODE_RAW) or \
                           (self.main_window.current_mode == self.main_window.MODE_NORMAL and self.hand_side == self.main_window.selected_hand)
            
            if text_visible:
                if finger_name == self.current_highlighted_finger and is_editable:
                    highlight_color = QColor(self.secondary_color);
                    highlight_color.setAlpha(150)
                    painter.setPen(Qt.NoPen);
                    painter.setBrush(highlight_color);
                    painter.setOpacity(0.9)
                    painter.drawRect(highlight_rect);
                    painter.setOpacity(1.0)
                
                keybind = self._humanize_key(keybind_raw)
                font = QFont();
                font.setBold(True);
                max_size, min_size = 14, 6
                for size in range(max_size, min_size - 1, -1):
                    font.setPointSize(size);
                    fm = QFontMetrics(font)
                    if fm.horizontalAdvance(keybind if keybind else " ") <= (w - 8): break
                painter.setFont(font);
                painter.setPen(QColor(255, 255, 255))
                
                display_text = keybind
                
                text_rect = QRect(x, y + int(h * 0.1), w, fm.height() + 6)
                painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignVCenter, display_text)
                
                if keybind_raw is not None and is_editable:
                    current_threshold = current_thresholds.get(config_key, 0.5)
                    threshold_text = f"T: {current_threshold:.2f}"
                    font.setPointSize(max(min_size, font.pointSize() - 2));
                    painter.setFont(font)
                    
                    # --- FIX START ---
                    # We access the slider locally using the internal key structure
                    slider_internal_key = f"{self.hand_side} Hand - {finger_name}"
                    
                    if slider_internal_key in self.sliders:
                        slider = self.sliders[slider_internal_key]['slider']
                        # We use the slider's geometry directly. If is_editable is True,
                        # the slider is effectively active, so we draw the text.
                        label_rect = QRect(slider.x(), slider.y() - QFontMetrics(font).height() - 2, slider.width(), QFontMetrics(font).height())
                        painter.drawText(label_rect, Qt.AlignHCenter | Qt.AlignVCenter, threshold_text)
                    # --- FIX END ---
    
    def _humanize_key(self, raw):
        if not raw: return ""
        return f"{raw.split('mouse:')[1].upper()} CLICK" if isinstance(raw, str) and raw.startswith("mouse:") else str(raw).upper()
    
    def mouseMoveEvent(self, event):
        finger_name = self._get_finger_at_pos(event.pos())
        self.current_highlighted_finger = finger_name if finger_name and self._is_editable(finger_name) else None
        self.update()
    
    def mousePressEvent(self, event):
        if isinstance(self.childAt(event.pos()), QSlider): super().mousePressEvent(event); return
        
        finger_name = self._get_finger_at_pos(event.pos())
        if not finger_name or not self._is_editable(finger_name): return
        
        config_key = self._get_key_for_mode(finger_name)
        mode_key, threshold_key = self.main_window.current_mode_key, self.main_window.current_threshold_key
        current_keybinds, current_thresholds = self.config.data.get(mode_key, {}), self.config.data.get(threshold_key, {})
        current_bind = current_keybinds.get(config_key)
        
        if event.button() == Qt.RightButton:
            if config_key in current_keybinds: current_keybinds[config_key] = None
            if config_key in current_thresholds: del current_thresholds[config_key]
            self.config.save_config();
            self.update_keybind_displays()
            print(f"Keybind and Threshold for {config_key} cleared by Right Click in {self.main_window.current_mode} Mode.")
            if self.key_assignment_window and self.key_assignment_window.isVisible(): self.key_assignment_window.close()
            return
        
        if event.button() == Qt.LeftButton:
            if self.key_assignment_window and self.key_assignment_window.isVisible(): self.key_assignment_window.close()
            self.key_assignment_window = KeyAssignmentWindow(
                finger_key=config_key, current_keybind=current_bind, config=self.config,
                primary_color=self.primary_color, secondary_color=self.secondary_color,
                tertiary_color=self.tertiary_color, parent=self.window()
            );
            self.key_assignment_window.show()
    
    def leaveEvent(self, event):
        self.current_highlighted_finger = None;
        self.update()
    
    def _get_finger_at_pos(self, pos):
        is_visible = (self.main_window.current_mode == self.main_window.MODE_RAW) or \
                     (self.main_window.current_mode == self.main_window.MODE_NORMAL and self.hand_side == self.main_window.selected_hand)
        if not is_visible: return None
        
        w_scale, h_scale = self.width() / 1000.0, self.height() / 1000.0
        norm_x, norm_y = pos.x() / w_scale, pos.y() / h_scale
        
        is_flipped = (self.main_window.current_mode == self.main_window.MODE_RAW and self.hand_side == "Left") or \
                     (self.main_window.current_mode == self.main_window.MODE_NORMAL and self.main_window.selected_hand == "Left")
        if is_flipped: norm_x = 1000 - norm_x
        
        for name, norm_rect in self.FINGER_REGIONS_NORM.items():
            if norm_rect.contains(QPoint(int(norm_x), int(norm_y))): return name
        return None


class KeybindsWindow(QWidget):
    DEFAULT_PRIMARY, DEFAULT_SECONDARY, DEFAULT_TERTIARY = "#44475a", "#50fa7b", "#ff5555"
    MODE_NORMAL, MODE_RAW = "Normal", "Raw"
    KEY_NORMAL, KEY_RAW = "n_keybinds", "r_keybinds"
    THRESHOLD_NORMAL, THRESHOLD_RAW = "n_thresholds", "r_thresholds"
    
    def __init__(self, config, primary_color=DEFAULT_PRIMARY, secondary_color=DEFAULT_SECONDARY, tertiary_color=DEFAULT_TERTIARY):
        super().__init__(None)
        self.config = config
        self.primary_color, self.secondary_color, self.tertiary_color = primary_color, secondary_color, tertiary_color
        self.setWindowTitle("Keybind Editor");
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setFixedSize(650, 550);
        self.layout = QVBoxLayout(self)
        self.current_mode, self.selected_hand = self.MODE_NORMAL, "Right"
        
        # This map helps the HandImageWidget find the correct, visible slider, regardless of the key used.
        self.key_map_raw_to_current_sliders = {}
        
        for key in [self.KEY_NORMAL, self.KEY_RAW, self.THRESHOLD_NORMAL, self.THRESHOLD_RAW]: self.config.data.setdefault(key, {})
        
        # --- UI Setup ---
        self._setup_control_bar()
        self._setup_header_and_instructions()
        self._setup_hand_widgets()
        
        self.setStyleSheet(self._get_base_style())
        self._update_mode_ui()
        QTimer.singleShot(100, self.initial_update)
    
    @property
    def current_mode_key(self):
        return self.KEY_NORMAL if self.current_mode == self.MODE_NORMAL else self.KEY_RAW
    
    @property
    def current_threshold_key(self):
        return self.THRESHOLD_NORMAL if self.current_mode == self.MODE_NORMAL else self.THRESHOLD_RAW
    
    def _toggle_mode(self):
        self.current_mode = self.MODE_RAW if self.current_mode == self.MODE_NORMAL else self.MODE_NORMAL
        self._update_mode_ui();
        self.update_keybind_displays()
    
    def _set_selected_hand(self, hand):
        if self.current_mode == self.MODE_NORMAL and self.selected_hand != hand:
            self.selected_hand = hand
            self._update_mode_ui();
            self.left_hand_widget.update_hand_display()
            self.right_hand_widget.update_hand_display();
            self.update_keybind_displays()
    
    def _setup_control_bar(self):
        self.control_bar_layout = QHBoxLayout()
        self.mode_toggle_button = QPushButton(f"Mode: {self.current_mode}");
        self.mode_toggle_button.clicked.connect(self._toggle_mode)
        self.mode_toggle_button.setFixedSize(120, 30);
        self.control_bar_layout.addWidget(self.mode_toggle_button);
        self.control_bar_layout.addStretch(1)
        
        self.hand_selector_layout = QHBoxLayout()
        self.left_hand_button = QPushButton("Left Hand");
        self.right_hand_button = QPushButton("Right Hand")
        self.left_hand_button.clicked.connect(lambda: self._set_selected_hand("Left"))
        self.right_hand_button.clicked.connect(lambda: self._set_selected_hand("Right"))
        self.hand_selector_layout.addWidget(self.left_hand_button);
        self.hand_selector_layout.addWidget(self.right_hand_button)
        
        self.control_bar_layout.addLayout(self.hand_selector_layout);
        self.control_bar_layout.addStretch(1)
        self.layout.addLayout(self.control_bar_layout)
    
    def _setup_header_and_instructions(self):
        header = QLabel("Click a Finger to Assign a Keybind");
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"font-size: 20px; font-weight: bold; padding: 10px; color: white;")
        self.layout.addWidget(header)
        
        instruction_layout = QVBoxLayout()
        clear_instruction = QLabel("Right click to clear the keybind");
        clear_instruction.setAlignment(Qt.AlignCenter)
        clear_instruction.setStyleSheet("font-size: 12px; color: #ffffff;")
        slider_instruction = QLabel("Adjust the threshold for how low each finger needs to be to trigger the action")
        slider_instruction.setAlignment(Qt.AlignCenter);
        slider_instruction.setStyleSheet("font-size: 12px; color: #ffffff; padding-bottom: 5px;")
        
        for widget in [clear_instruction, slider_instruction]: instruction_layout.addWidget(widget)
        self.layout.addLayout(instruction_layout)
    
    def _setup_hand_widgets(self):
        self.hand_layout = QHBoxLayout();
        self.hand_layout.setSpacing(20)
        
        hand_data = [("Left", "Left Hand (Secondary)"), ("Right", "Right Hand (Primary)")]
        for side, label_text in hand_data:
            container = QVBoxLayout()
            
            # Store the generic label text here. The full label text is set in _update_mode_ui
            # based on the current mode to avoid redundant hand labels in Normal Mode.
            label = QLabel(label_text);
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
            widget = HandImageWidget(hand_side=side, config=self.config, primary_color=self.primary_color,
                                     secondary_color=self.secondary_color, tertiary_color=self.tertiary_color, parent=self)
            
            container.addWidget(label);
            container.addWidget(widget);
            self.hand_layout.addLayout(container)
            setattr(self, f"{side.lower()}_hand_widget", widget)
            setattr(self, f"{side.lower()}_hand_label", label)
        
        self.layout.addLayout(self.hand_layout)
        note = QLabel("Note: Changes are saved instantly. Keybinds are captured globally.")
        note.setAlignment(Qt.AlignCenter);
        note.setStyleSheet("font-size: 10px; color: #FFFFFF; padding-top: 10px;")
        self.layout.addWidget(note)
    
    def _update_mode_ui(self):
        self.mode_toggle_button.setText(f"Mode: {self.current_mode}")
        is_normal_mode = self.current_mode == self.MODE_NORMAL
        
        self.left_hand_button.setVisible(is_normal_mode);
        self.right_hand_button.setVisible(is_normal_mode)
        
        if is_normal_mode:
            self.left_hand_widget.setVisible(self.selected_hand == "Left");
            self.left_hand_label.setVisible(self.selected_hand == "Left")
            self.right_hand_widget.setVisible(self.selected_hand == "Right");
            self.right_hand_label.setVisible(self.selected_hand == "Right")
            self.left_hand_button.setStyleSheet(self._get_hand_button_style(self.selected_hand == "Left"))
            self.right_hand_button.setStyleSheet(self._get_hand_button_style(self.selected_hand == "Right"))
        else:
            self.left_hand_widget.setVisible(True);
            self.left_hand_label.setVisible(True)
            self.right_hand_widget.setVisible(True);
            self.right_hand_label.setVisible(True)
        
        self.left_hand_widget.update_hand_display();
        self.right_hand_widget.update_hand_display()
    
    def _get_hand_button_style(self, is_selected):
        # --- LOGIC INVERTED HERE ---
        # Selected = Darker (Primary color)
        # Unselected = Lighter (Secondary color)
        
        if is_selected:
            bg_color = self.primary_color  # Darker
            text_color = "white"  # White text for contrast on dark
        else:
            bg_color = self.secondary_color  # Lighter
            text_color = self.primary_color  # Dark text on light
        
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: 2px solid {self.secondary_color};
                padding: 5px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #69ff9a;
                color: {self.primary_color};
            }}
        """
    
    def _get_base_style(self):
        # We ensure the default buttons (like the Mode button) use the
        # Lighter (Secondary) color so they look "clickable" (matching the unselected hands).
        return f"""
            QWidget {{ background-color: {self.primary_color}; color: white; border: 2px solid {self.tertiary_color}; border-radius: 12px; }}
            QLabel {{ color: white; background: transparent; }}
            QPushButton {{
                background-color: {self.secondary_color};
                color: {self.primary_color};
                border: none;
                padding: 5px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #69ff9a; }}
        """
    
    def initial_update(self):
        self.update_keybind_displays()
    
    def update_keybind_displays(self):
        self.key_map_raw_to_current_sliders.clear()
        
        # Build a temporary map of the currently visible and active sliders for use in paintEvent
        for hand_widget in [self.left_hand_widget, self.right_hand_widget]:
            if hand_widget.isVisible():
                for full_key_raw, data in hand_widget.sliders.items():
                    slider, finger_name = data['slider'], data['finger_name']
                    if slider.isVisible():
                        config_key = hand_widget._get_key_for_mode(finger_name)
                        self.key_map_raw_to_current_sliders[config_key] = slider
        
        self.right_hand_widget.update_keybind_displays();
        self.left_hand_widget.update_keybind_displays()


if __name__ == "__main__":
    class DummyConfig:
        def __init__(self):
            self.data = {
                "n_keybinds": {},
                "r_keybinds": {},
                "n_thresholds": {},
                "r_thresholds": {},
            }
        
        def save_config(self):
            print("\n--- CONFIG SAVED ---")
            for mode, config_data in sorted(self.data.items()):
                if mode.endswith("_keybinds"):
                    mode_prefix = mode.split("_")[0];
                    threshold_key = f"{mode_prefix}_thresholds"
                    thresholds = self.data.get(threshold_key, {})
                    print(f"\n--- {mode.upper()} & {threshold_key.upper()} ---")
                    for k, v in sorted(config_data.items()):
                        threshold = thresholds.get(k, "N/A")
                        key_pretty = self._humanize_key(v) if v else "None"
                        threshold_pretty = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else str(threshold)
                        print(f"KEY: {k:<25} | Bind: {key_pretty:<15} | Threshold: {threshold_pretty}")
            print("--------------------\n")
        
        def _humanize_key(self, raw):
            if not raw: return "None"
            return f"{raw.split('mouse:')[1].upper()} CLICK" if isinstance(raw, str) and raw.startswith("mouse:") else str(raw).upper()
    
    
    app = QApplication(sys.argv)
    cfg = DummyConfig()
    w = KeybindsWindow(cfg)
    w.show()
    sys.exit(app.exec_())