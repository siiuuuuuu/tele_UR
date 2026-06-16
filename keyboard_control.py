"""Keyboard control state for teleoperation collection."""

import threading

from pynput import keyboard


class KeyboardControl:
    """Tracks collection stop and episode recording toggles from keyboard input."""

    def __init__(self):
        self.collect_end = threading.Event()
        self.recording = threading.Event()
        self.listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                k = key.char.lower()
                if k == "a" and not self.collect_end.is_set():
                    self.collect_end.set()
                    print("\n[INFO] 检测到按键 'a'，结束数据采集循环...")
                elif (
                    k == "s"
                    and not self.collect_end.is_set()
                    and not self.recording.is_set()
                ):
                    self.recording.set()
                    print("\n[INFO] 检测到按键 's'，开始录制...")
                elif (
                    k == "s"
                    and not self.collect_end.is_set()
                    and self.recording.is_set()
                ):
                    self.recording.clear()
                    print("\n[INFO] 检测到按键 's'，结束录制...")
        except AttributeError:
            pass
        except Exception as e:
            print(f"Error in key press handler: {e}")

    def start(self):
        self.listener.start()

    def stop(self):
        if self.listener.running:
            self.listener.stop()

    def wait_recording(self, poll_interval=0.05):
        while not self.collect_end.is_set():
            if self.recording.wait(poll_interval):
                return True
        return False

    def is_recording(self):
        return self.recording.is_set()

    def clear_recording(self):
        self.recording.clear()

    def should_stop_collection(self):
        return self.collect_end.is_set()
