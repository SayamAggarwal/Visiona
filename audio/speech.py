"""
SpeechEngine — thread-safe TTS with priority queue and stale-message dropping.
"""
import threading
import queue
import time
import re
from datetime import datetime
from core.config import Config

try:
    import pyttsx3
    _PYTTSX3 = True
except ImportError:
    _PYTTSX3 = False
    print("  [Speech] pyttsx3 not installed.")

_MAX_AGE = 8.0   # ← restored to generous value — don't drop messages too fast


class SpeechEngine:

    def __init__(self):
        self._eq = queue.Queue()
        self._pq = queue.Queue(maxsize=5)   # ← slightly larger, still bounded
        self._nq = queue.Queue(maxsize=1)
        self.last_spoken_time = 0.0
        self.last_spoken_text = ""
        self._running = True
        self._volume  = 1.0
        self._interrupt_requested = False

        if _PYTTSX3:
            try:
                self.engine = pyttsx3.init(driverName='sapi5')
                self.engine.setProperty('rate', Config.SPEECH_RATE)
                self.engine.setProperty('volume', self._volume)
                for v in self.engine.getProperty('voices'):
                    if 'female' in v.name.lower() or 'zira' in v.name.lower():
                        self.engine.setProperty('voice', v.id)
                        break
            except Exception as e:
                print(f"  [Speech] Init warning: {e}")
                self.engine = None
        else:
            self.engine = None

        threading.Thread(target=self._worker, daemon=True).start()

    def duck(self):
        self._interrupt_requested = True
        self._volume = 0.0
        if self.engine:
            try:
                self.engine.setProperty('volume', 0.0)
            except Exception:
                pass

    def unduck(self):
        self._interrupt_requested = False
        self._volume = 1.0
        if self.engine:
            try:
                self.engine.setProperty('volume', 1.0)
            except Exception:
                pass

    def interrupt(self):
        self._flush_all()

    def _worker(self):
        while self._running:
            item = None
            try:
                # Emergency always first
                item = self._eq.get_nowait()
            except queue.Empty:
                if not self._interrupt_requested:
                    try:
                        item = self._pq.get_nowait()
                    except queue.Empty:
                        try:
                            item = self._nq.get(timeout=0.05)
                        except queue.Empty:
                            pass
                else:
                    time.sleep(0.05)

            if item is None:
                continue

            # Handle sentinel for shutdown
            if not isinstance(item, tuple):
                continue

            text, ts, is_emergency, scheduled_time = item

            # Drop stale messages — but only non-emergency ones
            if not is_emergency and (time.time() - ts) > _MAX_AGE:
                continue

            if not is_emergency and scheduled_time is not None:
                now = time.time()
                if now - scheduled_time > _MAX_AGE:
                    continue
                if scheduled_time > now:
                    time.sleep(scheduled_time - now)

            try:
                with open("conversation_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"SPEECH ({datetime.now().strftime('%H:%M:%S')}): {text}\n")
            except Exception:
                pass

            if self.engine:
                try:
                    target_vol = 1.0 if is_emergency else self._volume
                    self.engine.setProperty('volume', target_vol)
                    self.engine.say(text)
                    self.engine.runAndWait()
                    if is_emergency:
                        self.engine.setProperty('volume', self._volume)
                    self.last_spoken_time = time.time()
                except Exception as e:
                    print(f"  [Speech] Playback error: {e}")
            else:
                print(f"  [Speech] {text}")

    def _flush_all(self):
        for q in (self._eq, self._pq, self._nq):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

    def _flush_normal(self):
        while not self._nq.empty():
            try: self._nq.get_nowait()
            except queue.Empty: break

    def speak(self, text: str, priority: bool = False, bypass_cooldown: bool = False,
              emergency: bool = False, scheduled_time: float = None):
        if not text:
            return
        now = time.time()

        semantic_base = re.sub(r'\d+', '', text.strip().lower())

        if not hasattr(self, '_semantic_history'):
            self._semantic_history = {}

        last_time = self._semantic_history.get(semantic_base, 0)
        is_warning = any(
            w in semantic_base
            for w in ("warning", "very close", "found", "emergency")
        ) or emergency
        required_cooldown = (
            Config.SPEECH_COOLDOWN if is_warning
            else getattr(Config, 'SEMANTIC_COOLDOWN', 12.0)
        )
        cooldown_ok = bypass_cooldown or emergency or (now - last_time) > required_cooldown

        if emergency:
            self._eq.put((text, now, True, None))
            self._semantic_history[semantic_base] = now
            self.last_spoken_time = now

        elif priority:
            if not bypass_cooldown:
                self._flush_normal()
            try:
                self._pq.put_nowait((text, now, False, scheduled_time))
            except queue.Full:
                try: self._pq.get_nowait()   # drop oldest to make room
                except queue.Empty: pass
                try: self._pq.put_nowait((text, now, False, scheduled_time))
                except queue.Full: pass
            self._semantic_history[semantic_base] = now
            self.last_spoken_time = now

        elif cooldown_ok:
            self._flush_normal()
            try:
                self._nq.put_nowait((text, now, False, scheduled_time))
                self._semantic_history[semantic_base] = now
            except queue.Full:
                pass

    def speak_all(self, messages: list, first_priority: bool = False):
        # ← Restored to original MAX_MESSAGES limit, no artificial cap
        # Only speak the single most important message per cycle
        # unless there's a high priority threat, then allow 2
        limit = 2 if first_priority else 1
        for i, msg in enumerate(messages[:limit]):
            self.speak(msg, priority=(i == 0 and first_priority))

    def stop(self):
        self._running = False