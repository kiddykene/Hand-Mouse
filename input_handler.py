import keyboard

class InputHandler:
    def __init__(self, capture_key='f8'):
        self.capture_key = capture_key
        self.suppress_capture = False
        self.capture_requested = False
        keyboard.on_press(self._on_any_key)

    def _on_any_key(self, event):
        try:
            if getattr(self, "suppress_capture", False):
                return
            name = getattr(event, "name", None)
            if name == self.capture_key:
                self.capture_requested = True
        except Exception:
            pass

    def trigger_capture(self):
        self.capture_requested = True

    def set_keybind(self, new_key):
        if isinstance(new_key, str) and new_key:
            self.capture_key = new_key
