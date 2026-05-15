"""
Visiona AI — Multi-camera assistive navigation system.

Architecture:
  4 CameraFeed threads → VisionSystem (YOLO + depth + track + speed + TTC)
  → DetectionPriorityQueue → grouping → SpeechEngine + AlertSystem
"""
import cv2
import os
import time
import threading
import numpy as np
from typing import Dict, List, Optional
from pynput import keyboard

from core.config import Config
from core.detection import Detection
from core.priority_queue import DetectionPriorityQueue
from perception.vision import VisionSystem
from kinematics.heatmap import group_detections, build_speech_messages
from audio.alert import AlertSystem
from audio.speech import SpeechEngine
from audio.voice_input import VoiceInputEngine
from core.logger import SessionLogger

from agents.orchestrator import AgentEngine
from core.memory import memory_bank
from core.recognition import feature_db


class CameraFeed:
    """Manages a single camera/video source for one direction synchronously."""

    def __init__(self, direction: str, source):
        self.direction = direction
        self.cap: Optional[cv2.VideoCapture] = None
        self.active = False
        self.source = source
        self.fps = 30.0
        self.frame_time = 1.0 / 30.0

        if source is None:
            return

        self._open_capture(source)

    def _open_capture(self, source):
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print(f"  [Camera] Could not open {self.direction} source: {source}")
                if isinstance(source, int) and source != 0:
                    print(f"  [Camera] Attempting fallback to device 0 for {self.direction}...")
                    self.cap = cv2.VideoCapture(0)

                if not self.cap or not self.cap.isOpened():
                    self.cap = None
                    return

            if isinstance(source, str):
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0 or self.fps > 120:
                    self.fps = 30.0
                self.frame_time = 1.0 / self.fps
                print(f"  [Camera] {self.direction} feed active (Source: {source}, FPS: {self.fps:.1f})")
            else:
                print(f"  [Camera] {self.direction} feed active (Source: {source})")

            self.active = True
        except Exception as e:
            print(f"  [Camera] Error opening {self.direction}: {e}")
            self.cap = None

    def get_frame(self):
        if not self.active or not self.cap:
            return None
        try:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str) and not self.source.isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret:
                    self.active = False
                    return None
            return frame
        except Exception as e:
            print(f"  [Camera] Runtime error reading {self.direction}: {e}")
            return None

    def release(self):
        self.active = False
        if self.cap:
            self.cap.release()


class VisionaApp:

    def __init__(self):
        print("\n  Visiona AI — Starting up...\n")

        self.vision = VisionSystem()
        self.alert  = AlertSystem()
        self.speech = SpeechEngine()
        self.logger = SessionLogger()
        self.agent_engine = AgentEngine(
            tts_callback=lambda x: self.speech.speak(
                x, priority=True, bypass_cooldown=True, emergency=True
            ),
            search_intent_callback=self._on_intent,
        )

        self.feeds: Dict[str, CameraFeed] = {
            d: CameraFeed(d, s) for d, s in Config.SOURCES.items()
        }
        active = [d for d, f in self.feeds.items() if f.active]
        if not active:
            raise RuntimeError("No active camera feeds. Check Config.SOURCES.")
        print(f"\n  Active feeds: {', '.join(active)}")

        self._frame_count   = 0
        self._all_dets: List[Detection] = []
        self._state         = "SCANNING"
        self._last_info     = ""
        self._mic_active    = False
        self._mic_mode      = "LLM"
        self._search_intent: Optional[str] = None
        self._search_intents: List[str] = []
        self._running       = True
        self._latest_frames: Dict[str, np.ndarray] = {}
        self._curiosity_cooldown = 0.0

        self._capture_alias      = None
        self._capture_base_class = None
        self._capture_count      = 0

        # Runtime state dicts — initialised here to avoid scattered hasattr checks
        self._universal_tracker: Dict  = {}
        self._ambient_states: Dict     = {}
        self._goal_states: Dict        = {}
        self._crowd_alert_time: float  = 0.0
        self._emergency_alert_times: Dict = {}

        os.makedirs("database", exist_ok=True)

        self._ai_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._ai_thread.start()

        self.voice = VoiceInputEngine(
            on_speech=self._on_speech,
            on_listening=self._on_listening,
        )
        self.voice.start()

        self.speech.speak("Visiona AI ready.")
        print("\n  [System] Visiona AI Initialized.")
        print("  [System] V (Hold) = General AI reasoning / needs.")
        print("  [System] G (Hold) = Dedicated GPS Navigation.")
        print("  [System] R (Hold) = Memorize object.")
        print("  [System] Controls: ESC = quit\n")

        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._keyboard_listener.start()

    # ------------------------------------------------------------------
    # Keyboard PTT
    # ------------------------------------------------------------------

    def _on_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char == 'v' and not self._mic_active:
                    self._mic_mode = "LLM"
                    self.voice.start_recording()
                elif key.char == 'g' and not self._mic_active:
                    self._mic_mode = "MAPS"
                    self.voice.start_recording()
                elif key.char and key.char.lower() == 'r' and not self._mic_active:
                    self._mic_mode = "REMEMBER"
                    self.voice.start_recording()
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char == 'v' and self._mic_mode == "LLM":
                    self.voice.stop_recording()
                elif key.char == 'g' and self._mic_mode == "MAPS":
                    self.voice.stop_recording()
                elif key.char and key.char.lower() == 'r' and self._mic_mode == "REMEMBER":
                    self.voice.stop_recording()
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Main display loop
    # ------------------------------------------------------------------

    def run(self):
        cv2.namedWindow("Visiona AI - Unified HUD", cv2.WINDOW_NORMAL)

        target_fps = 30.0
        for feed in self.feeds.values():
            if feed.active:
                target_fps = feed.fps
                break

        frame_delay = int(1000.0 / target_fps)
        print(f"  [System] Video playback at {target_fps:.1f} FPS (frame delay: {frame_delay}ms)")

        while self._running:
            frame_start = time.time()
            self._frame_count += 1
            frames_to_render = {}

            # 1. Read all frames
            raw_frames = {}
            for direction, feed in self.feeds.items():
                if feed.active:
                    frame = feed.get_frame()
                    if frame is not None:
                        raw_frames[direction] = cv2.resize(
                            frame, (Config.DISPLAY_W, Config.DISPLAY_H)
                        )

            if not raw_frames:
                break

            # Hand frames to background AI thread
            self._latest_frames = raw_frames.copy()

            # 2. Draw overlay
            for direction, frame in raw_frames.items():
                self.vision.draw_overlay(
                    frame,
                    [d for d in self._all_dets if d.direction == direction],
                    self._state,
                    self._last_info,
                )
                self._draw_extras(frame, direction)
                frames_to_render[direction] = frame

            # 3. Build grid and show
            grid_frame = self._build_grid(frames_to_render)
            cv2.imshow("Visiona AI - Unified HUD", grid_frame)

            # 4. Precise FPS timing
            elapsed = (time.time() - frame_start) * 1000
            wait_time = max(1, int(frame_delay - elapsed))
            key = cv2.waitKey(wait_time) & 0xFF
            if key == 27:
                break

        self._shutdown()

    # ------------------------------------------------------------------
    # Background AI processing loop
    # ------------------------------------------------------------------

    def _processing_loop(self):
        """YOLO + depth evaluation at controlled rate."""
        frame_counter = 0

        while self._running:
            start_t = time.time()
            frames = dict(self._latest_frames)

            if not frames:
                time.sleep(0.01)
                continue

            frame_counter += 1

            if frame_counter % Config.FRAME_SKIP != 0:
                time.sleep(0.01)
                continue

            all_dets = []
            for direction, frame in frames.items():
                dets = self.vision.detect(frame, direction)

                # Feature DB matching — only every 5th processed frame (expensive)
                fh, fw = frame.shape[:2]
                if frame_counter % 5 == 0:
                    for d in dets:
                        x1, y1, x2, y2 = d.box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(fw, x2), min(fh, y2)
                        w, h = x2 - x1, y2 - y1
                        if w > 40 and h > 40:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                matched_alias = feature_db.match(crop)
                                if matched_alias:
                                    d.base_label = d.label
                                    d.label = matched_alias

                all_dets.extend(dets)
                self.logger.log_detections(dets, direction)

                # Capture logic for memorisation
                if (
                    self._capture_count > 0
                    and self._capture_alias
                    and self._capture_base_class
                    and dets
                ):
                    target_dets = [
                        d for d in dets
                        if d.label.lower() == self._capture_base_class.lower()
                    ]
                    if target_dets:
                        best_det = max(target_dets, key=lambda x: x.threat_score)
                        x1, y1, x2, y2 = best_det.box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(fw, x2), min(fh, y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            folder = os.path.join("database", self._capture_alias)
                            os.makedirs(folder, exist_ok=True)
                            filepath = os.path.join(
                                folder, f"img_{50 - self._capture_count}.jpg"
                            )
                            cv2.imwrite(filepath, crop)
                            self._capture_count -= 1
                            if self._capture_count == 0:
                                print(
                                    f"  [Capture] Finished capturing 50 frames for "
                                    f"{self._capture_alias}"
                                )
                                self.speech.speak(
                                    f"Finished capturing images for "
                                    f"{self._capture_alias}. Connecting memory bank.",
                                    bypass_cooldown=True,
                                    emergency=True,
                                )
                                feature_db.load_alias(self._capture_alias)
                                self._capture_alias = None
                                self._capture_base_class = None

            if all_dets:
                self._all_dets = all_dets
                self._pipeline(all_dets)
            else:
                self._all_dets = []

            # Adaptive sleep — target ~6.7 FPS processing rate
            elapsed = time.time() - start_t
            target_cycle_time = 0.10
            if elapsed < target_cycle_time:
                time.sleep(target_cycle_time - elapsed)

    # ------------------------------------------------------------------
    # Grid builder
    # ------------------------------------------------------------------

    def _build_grid(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        count = len(frames)
        if count == 0:
            return np.zeros((Config.DISPLAY_H, Config.DISPLAY_W, 3), dtype=np.uint8)

        blank = np.zeros((Config.DISPLAY_H, Config.DISPLAY_W, 3), dtype=np.uint8)
        cv2.putText(
            blank, "NO SIGNAL",
            (Config.DISPLAY_W // 2 - 60, Config.DISPLAY_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA,
        )

        if count == 1:
            return list(frames.values())[0]
        elif count == 2:
            keys = list(frames.keys())
            grid = np.hstack((frames[keys[0]], frames[keys[1]]))
            return cv2.resize(grid, (Config.DISPLAY_W * 2, Config.DISPLAY_H))
        else:
            front = frames.get("FRONT", blank)
            left  = frames.get("LEFT",  blank)
            right = frames.get("RIGHT", blank)
            back  = frames.get("BACK",  blank)
            top    = np.hstack((front, right))
            bottom = np.hstack((left,  back))
            grid   = np.vstack((top, bottom))
            target_w = int(Config.DISPLAY_W * 1.5)
            target_h = int(Config.DISPLAY_H * 1.5)
            return cv2.resize(grid, (target_w, target_h))

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------

    def _pipeline(self, detections: List[Detection]):
        pq = DetectionPriorityQueue()
        pq.push_all(detections)
        sorted_dets = pq.drain()

        now = time.time()

        # Universal strict distance tracker
        for lbl in list(self._universal_tracker.keys()):
            if now - self._universal_tracker[lbl]["last_seen"] > 15.0:
                self._universal_tracker[lbl]["min_steps"] = 999

        dets_by_dist = sorted(
            sorted_dets,
            key=lambda x: x.distance_m if x.distance_m is not None else 999.0,
        )

        for d in dets_by_dist:
            d.is_new_record_distance = False
            label_lower = d.label.lower()
            d_val = d.distance_m if d.distance_m is not None else 999.0
            steps = max(1, round(d_val / Config.METERS_PER_STEP)) if d_val != 999.0 else 999

            if label_lower not in self._universal_tracker:
                self._universal_tracker[label_lower] = {
                    "min_steps": 999, "last_seen": 0.0, "last_record_time": 0.0
                }

            state = self._universal_tracker[label_lower]
            state["last_seen"] = now

            if steps < state["min_steps"]:
                state["min_steps"] = steps
                state["last_record_time"] = now
                d.is_new_record_distance = True
            elif steps == state["min_steps"] and state["last_record_time"] == now:
                d.is_new_record_distance = True

        hp = [d for d in sorted_dets if d.is_high_priority]

        # State
        if hp:
            self._state = "ALERT"
        elif any(d.distance_m and d.distance_m < 3.0 for d in sorted_dets):
            self._state = "AVOIDING"
        else:
            self._state = "SCANNING"

        # Busy road
        vehicles = [
            d for d in detections
            if d.label in ("car", "truck", "bus", "motorcycle")
        ]
        if len(vehicles) >= 5:
            self.speech.speak(
                "Busy road detected. Use caution while navigating.", priority=True
            )
            self._state = "CAUTION"

        # ------ Ambient filter helpers ------

        from core.memory import goal_system
        goal_candidates = goal_system.get_active_candidates()
        search_intents = list(self._search_intents)
        if self._search_intent and self._search_intent not in search_intents:
            search_intents.append(self._search_intent)

        def _is_ambient_reportable(det: Detection) -> bool:
            if det.direction == "FRONT":
                return True
            label_lower = det.label.lower()
            is_goal = label_lower in [c.lower() for c in goal_candidates]
            is_search = any(
                i and i.lower() in label_lower for i in search_intents
            )
            base_l = getattr(det, 'base_label', det.label).lower()
            is_vehicle = base_l in (
                "car", "truck", "bus", "motorcycle", "bicycle"
            )
            return is_vehicle or is_goal or is_search

        def _is_threat(det: Detection) -> bool:
            base_l = getattr(det, 'base_label', det.label).lower()
            if det.ttc_sec is not None and det.ttc_sec <= Config.TTC_WARN_THRESHOLD:
                return True
            if base_l in ("car", "truck", "bus", "motorcycle", "bicycle") and \
                    getattr(det, 'motion', None) == "approaching":
                return True
            if det.distance_m is not None and det.distance_m < 1.0:
                if base_l in ("person", "car", "truck", "bus", "motorcycle", "bicycle"):
                    return True
            if det.threat_score > Config.THREAT_HIGH_THRESHOLD:
                if getattr(det, 'motion', None) == "approaching":
                    return True
            return False

        # Build ambient list
        raw_ambient_dets = [d for d in sorted_dets if _is_ambient_reportable(d)]
        ambient_dets = []
        allowed_labels_this_frame: set = set()

        for d in raw_ambient_dets:
            if _is_threat(d):
                ambient_dets.append(d)
                continue

            label_lower = d.label.lower()
            dir_str = d.direction

            is_goal_or_search = label_lower in [c.lower() for c in goal_candidates] or \
                any(i and i.lower() in label_lower for i in search_intents)

            if is_goal_or_search:
                continue  # handled by _match_goals / _seek

            if label_lower not in self._ambient_states:
                self._ambient_states[label_lower] = {}

            key = f"{label_lower}_{dir_str}"
            if key in allowed_labels_this_frame:
                ambient_dets.append(d)
                continue

            cooldown_time = 7.0 if dir_str == "FRONT" else 3.0
            if now - self._ambient_states[label_lower].get(dir_str, 0.0) >= cooldown_time:
                ambient_dets.append(d)
                self._ambient_states[label_lower][dir_str] = now
                allowed_labels_this_frame.add(key)

        ambient_hp = [d for d in ambient_dets if d.is_high_priority]

        # Crowd detection
        person_count = sum(
            1 for d in sorted_dets
            if getattr(d, 'base_label', d.label).lower() == "person"
            and d.direction == "FRONT"
        )
        if person_count > 5 and (now - self._crowd_alert_time) >= 60.0:
            self.speech.speak(
                "You are standing in a crowded area where so many people are walking "
                "towards you. So please be careful.",
                priority=True, emergency=True,
            )
            self._crowd_alert_time = now

        if (now - self._crowd_alert_time) < 60.0:
            ambient_dets = [
                d for d in ambient_dets
                if getattr(d, 'base_label', d.label).lower() != "person"
            ]
            ambient_hp = [
                d for d in ambient_hp
                if getattr(d, 'base_label', d.label).lower() != "person"
            ]

        # Emergency verbal bypass
        for d in ambient_hp:
            base_l = getattr(d, 'base_label', d.label).lower()
            dir_s = {
                "FRONT": "ahead", "LEFT": "on the left",
                "RIGHT": "on the right", "BACK": "behind you",
            }.get(d.direction, "nearby")
            emergency_key = f"{base_l}_{d.direction}"
            if now - self._emergency_alert_times.get(emergency_key, 0.0) < 3.0:
                continue
            if base_l in ("truck", "bus", "car", "motorcycle", "bicycle") \
                    and d.distance_m and d.distance_m < 2.5:
                self.speech.speak(f"Emergency: {d.label} {dir_s}!", emergency=True)
                self._emergency_alert_times[emergency_key] = now
            elif base_l == "person" and d.distance_m and d.distance_m < 1.0:
                self.speech.speak(f"{d.label} very close {dir_s}!", emergency=True)
                self._emergency_alert_times[emergency_key] = now

        # Beep
        self.alert.process(ambient_hp)

        # Speech summaries
        grouped  = group_detections(ambient_dets)
        messages = build_speech_messages(grouped, ambient_hp)

        # Memory push
        full_context = self._get_full_spatial_context()
        memory_bank.add_detections([full_context])

        # Goals
        self._match_goals(sorted_dets)

        if messages:
            self._last_info = " | ".join(messages)
            self.logger.log_speech(messages)
            self.speech.speak_all(messages, first_priority=bool(ambient_hp))

    # ------------------------------------------------------------------
    # Goal evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_goal_object(self, det: Detection, label: str) -> str:
        label = label.lower()
        now = time.time()
        current_distance = det.distance_m if det.distance_m is not None else 999.0
        current_steps = (
            max(1, round(current_distance / Config.METERS_PER_STEP))
            if current_distance != 999.0 else 999
        )

        if label not in self._goal_states:
            self._goal_states[label] = {
                "record_steps": 999,
                "last_announced_time": 0.0,
                "has_repeated": True,
            }

        state = self._goal_states[label]

        if current_steps < state["record_steps"]:
            if now - state["last_announced_time"] >= 1.0 or state["record_steps"] == 999:
                print(
                    f"  [Goal] {label} NEW RECORD "
                    f"{state['record_steps']} → {current_steps} steps"
                )
                state["record_steps"] = current_steps
                state["last_announced_time"] = now
                state["has_repeated"] = False
                return "ANNOUNCE_NEW"

        if not state["has_repeated"]:
            if now - state["last_announced_time"] >= 5.0:
                print(f"  [Goal] {label} REPEAT at {state['record_steps']} steps")
                state["has_repeated"] = True
                state["last_announced_time"] = now
                return "ANNOUNCE_REPEAT"

        return "IGNORE"

    def _evaluate_target(self, det: Detection, label: str) -> str:
        return self._evaluate_goal_object(det, label)

    def _trigger_contextual_arrival(self, label: str, dir_s: str):
        if self.agent_engine and self.agent_engine.llm:
            from langchain_core.messages import HumanMessage

            def _ask_llm():
                prompt = (
                    f"The user is blind and has just arrived right next to a '{label}' "
                    f"({dir_s}). Write exactly ONE comforting, concise sentence telling "
                    "them they have reached the destination, suggesting what they can "
                    "logically do with it (like sit down if it's a chair), and asking "
                    "if they want to set a new goal. Do not use markdown."
                )
                try:
                    resp = self.agent_engine.llm.invoke([HumanMessage(content=prompt)])
                    text = resp.content if isinstance(resp.content, str) else str(resp.content)
                    self.speech.speak(
                        text, priority=True, bypass_cooldown=True, emergency=True
                    )
                except Exception as e:
                    print(f"  [Agent] Contextual fallback needed: {e}")
                    self.speech.speak(
                        f"You have reached the destination. The {label} is {dir_s}. "
                        "You can interact with it now.",
                        priority=True, bypass_cooldown=True, emergency=True,
                    )

            threading.Thread(target=_ask_llm, daemon=True).start()
        else:
            self.speech.speak(
                f"You have reached the destination. The {label} is {dir_s}. "
                "You can interact with it now.",
                priority=True, bypass_cooldown=True, emergency=True,
            )

    # ------------------------------------------------------------------
    # Goal matching
    # ------------------------------------------------------------------

    def _match_goals(self, detections: List[Detection]):
        from core.memory import goal_system
        from collections import defaultdict

        candidates = goal_system.get_active_candidates()
        if not candidates:
            return

        sorted_dets = sorted(
            detections,
            key=lambda x: x.distance_m if x.distance_m is not None else 999.0,
        )
        goal_objects: Dict = defaultdict(list)

        for d in sorted_dets:
            label_lower = d.label.lower()
            if label_lower in [c.lower() for c in candidates]:
                if d.distance_m is not None and d.distance_m <= 1.0:
                    dir_s = {
                        "FRONT": "directly in front of you",
                        "LEFT":  "right beside you on the left",
                        "RIGHT": "right beside you on the right",
                        "BACK":  "right behind you",
                    }.get(d.direction, "nearby")
                    goal_system.complete_goal(label_lower)
                    self._trigger_contextual_arrival(label_lower, dir_s)
                    if label_lower not in self._ambient_states:
                        self._ambient_states[label_lower] = {}
                    for direct in ("FRONT", "LEFT", "RIGHT", "BACK"):
                        self._ambient_states[label_lower][direct] = time.time() + 60.0
                    return
                goal_objects[label_lower].append(d)

        for label_lower, all_dets in goal_objects.items():
            if not all_dets:
                continue
            nearest = min(
                all_dets,
                key=lambda x: x.distance_m if x.distance_m is not None else 999.0,
            )
            eval_cmd = self._evaluate_goal_object(nearest, label_lower)
            if eval_cmd in ("ANNOUNCE_NEW", "ANNOUNCE_REPEAT"):
                direction  = nearest.direction
                distance_m = nearest.distance_m
                dist = (
                    f"at {max(1, round(distance_m / Config.METERS_PER_STEP))} "
                    f"step{'s' if round(distance_m / Config.METERS_PER_STEP) != 1 else ''}"
                    if distance_m else "nearby"
                )
                speak_dir = {
                    "FRONT": "in front", "LEFT": "on the left",
                    "RIGHT": "on the right", "BACK": "behind you",
                }.get(direction, "")
                count  = len(all_dets)
                plural = label_lower + "s" if not label_lower.endswith("s") else label_lower
                msg = (
                    f"Found a group of {plural}, {dist} {speak_dir}."
                    if count > 1 else
                    f"Found a {label_lower}, {dist} {speak_dir}."
                )
                self.speech.speak(msg, priority=True, bypass_cooldown=True)
                break

        if self._search_intent or self._search_intents:
            self._seek(detections)

    def _seek(self, detections: List[Detection]):
        from collections import defaultdict

        all_intents = list(self._search_intents)
        if self._search_intent and self._search_intent not in all_intents:
            all_intents.append(self._search_intent)

        if not all_intents:
            return

        for intent in all_intents:
            if not intent:
                continue

            matches = [d for d in detections if intent.lower() in d.label.lower()]
            if not matches:
                continue

            matches.sort(
                key=lambda x: x.distance_m if x.distance_m is not None else 999.0
            )

            grouped_matches: Dict = defaultdict(list)
            intent_completed = False

            for m in matches:
                if m.distance_m is not None and m.distance_m <= 1.0 and not intent_completed:
                    dir_s = {
                        "FRONT": "directly in front of you",
                        "LEFT":  "right beside you on the left",
                        "RIGHT": "right beside you on the right",
                        "BACK":  "right behind you",
                    }.get(m.direction, "nearby")
                    if intent in self._search_intents:
                        self._search_intents.remove(intent)
                    if intent == self._search_intent:
                        self._search_intent = None
                    self._state = "GUIDING"
                    self._trigger_contextual_arrival(m.label, dir_s)
                    intent_completed = True
                    label_lower = m.label.lower()
                    if label_lower not in self._ambient_states:
                        self._ambient_states[label_lower] = {}
                    for direct in ("FRONT", "LEFT", "RIGHT", "BACK"):
                        self._ambient_states[label_lower][direct] = time.time() + 60.0
                grouped_matches[m.direction].append(m)

            if intent_completed:
                return

            for direction, group in grouped_matches.items():
                t = group[0]
                eval_cmd = self._evaluate_target(t, intent)
                if eval_cmd in ("ANNOUNCE_NEW", "ANNOUNCE_REPEAT"):
                    count = len(group)
                    label_lower = t.label.lower()
                    plural = (
                        label_lower + "s"
                        if not label_lower.endswith("s") else label_lower
                    )
                    dist = (
                        f"at {max(1, round(t.distance_m / Config.METERS_PER_STEP))} "
                        f"step{'s' if round(t.distance_m / Config.METERS_PER_STEP) != 1 else ''}"
                        if t.distance_m else "nearby"
                    )
                    speak_dir = {
                        "FRONT": "in front", "LEFT": "on the left",
                        "RIGHT": "on the right", "BACK": "behind you",
                    }.get(direction, "")
                    msg = (
                        f"Found a group of {plural}, {dist} {speak_dir}."
                        if count > 1 else
                        f"Found a {label_lower}, {dist} {speak_dir}."
                    )
                    self.speech.speak(msg, priority=True, bypass_cooldown=True)
                    return

    # ------------------------------------------------------------------
    # Voice callbacks
    # ------------------------------------------------------------------

    def _on_listening(self, active: bool):
        self._mic_active = active
        if active:
            self.speech.duck()
            self.alert.pause()
        else:
            self.speech.unduck()
            self.alert.resume()

    def _on_speech(self, text: str):
        print(f"  [Voice] Received Speech ({self._mic_mode}): \"{text}\"")
        full_context = self._get_full_spatial_context()

        if self._mic_mode == "MAPS":
            if self.agent_engine.llm_with_tools:
                threading.Thread(
                    target=self.agent_engine.process_voice_command,
                    args=(f"I need walking directions to: {text}", full_context),
                    daemon=True,
                ).start()
            return

        if self._mic_mode == "REMEMBER":
            def _extract_and_capture():
                if self.agent_engine.llm:
                    data = self.agent_engine.extract_memory_label(text)
                    if data and isinstance(data, dict):
                        alias      = data.get("alias")
                        base_class = data.get("base_class")
                        if (
                            alias and alias.lower() not in ("none", "unknown")
                            and base_class
                        ):
                            self.speech.speak(
                                f"Okay, I will memorize {alias} now. Keep looking at it.",
                                bypass_cooldown=True, emergency=True,
                            )
                            time.sleep(2)
                            self._capture_alias      = alias
                            self._capture_base_class = base_class
                            self._capture_count      = 50
                        else:
                            self.speech.speak(
                                "I didn't catch what you wanted me to remember.",
                                bypass_cooldown=True, emergency=True,
                            )
                    else:
                        self.speech.speak(
                            "I didn't catch what you wanted me to remember.",
                            bypass_cooldown=True, emergency=True,
                        )
                else:
                    self.speech.speak(
                        "The reasoning engine is offline.",
                        bypass_cooldown=True, emergency=True,
                    )

            threading.Thread(target=_extract_and_capture, daemon=True).start()
            return

        if self.agent_engine.llm_with_tools:
            threading.Thread(
                target=self.agent_engine.process_voice_command,
                args=(text, full_context),
                daemon=True,
            ).start()
        else:
            self.speech.speak(
                f"I heard you say: {text}. Reasoning engine is offline.",
                bypass_cooldown=True,
            )

    def _on_intent(self, intent: str):
        print(f"  [Voice] Tool Triggered Search: {intent}")

        if self._all_dets:
            sorted_dets = sorted(
                self._all_dets,
                key=lambda x: x.distance_m if x.distance_m is not None else 999.0,
            )
            for d in sorted_dets:
                if intent.lower() in d.label.lower():
                    dist = (
                        f"{max(1, round(d.distance_m / Config.METERS_PER_STEP))} "
                        f"step{'s' if round(d.distance_m / Config.METERS_PER_STEP) != 1 else ''}"
                        if d.distance_m else "nearby"
                    )
                    dir_s = {
                        "FRONT": "in front", "LEFT": "on the left",
                        "RIGHT": "on the right", "BACK": "behind you",
                    }.get(d.direction, "")
                    self.speech.speak(
                        f"There's a {d.label} {dist} {dir_s}.",
                        priority=True, bypass_cooldown=True,
                    )
                    return

        if intent not in self._search_intents:
            self._search_intents.append(intent)
        self._search_intent = intent
        self._state = "SEARCHING"

    # ------------------------------------------------------------------
    # Spatial context for LLM
    # ------------------------------------------------------------------

    def _get_full_spatial_context(self) -> str:
        motion_ctx = (
            f"User State: {getattr(self.vision, 'user_state', 'Stationary')} "
            f"({getattr(self.vision, 'user_speed', 0.0):.1f} m/s)."
        )
        if not self._all_dets:
            return f"{motion_ctx} No objects currently detected in view."

        sorted_dets = sorted(
            self._all_dets,
            key=lambda x: x.distance_m if x.distance_m is not None else 999.0,
        )
        ctx_parts = []
        for d in sorted_dets:
            dist = (
                f"{max(1, round(d.distance_m / Config.METERS_PER_STEP))} steps"
                if d.distance_m else "unknown distance"
            )
            dir_s = {
                "FRONT": "in front", "LEFT": "on the left",
                "RIGHT": "on the right", "BACK": "behind you",
            }.get(d.direction, d.direction.lower())
            mot_s = f"({d.motion})" if getattr(d, 'motion', None) else ""
            ctx_parts.append(f"{d.label} at {dist} {dir_s} {mot_s}".strip())

        return f"{motion_ctx} Objects: " + " | ".join(ctx_parts)

    # ------------------------------------------------------------------
    # HUD extras
    # ------------------------------------------------------------------

    def _draw_extras(self, frame, direction: str):
        cv2.putText(
            frame, direction,
            (Config.DISPLAY_W // 2 - 30, Config.DISPLAY_H - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA,
        )
        color = (0, 220, 80) if self._mic_active else (80, 80, 80)
        cv2.circle(
            frame, (Config.DISPLAY_W - 18, Config.DISPLAY_H - 28),
            6, color, -1, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        self._running = False
        print("\n  [System] Shutting down Visiona AI...")
        stats = self.logger.get_stats()
        print(f"  [System] Session Statistics:")
        print(f"           - Duration: {stats['duration_s']}s")
        print(f"           - Total Events: {stats['events']}")
        print(
            f"           - Logs saved to: "
            f"{self.logger._path if hasattr(self.logger, '_path') else 'N/A'}"
        )
        self.voice.stop()
        for f in self.feeds.values():
            f.release()
        cv2.destroyAllWindows()
        self.speech.stop()
        print("  [System] Shutdown complete. Stay safe.")


if __name__ == "__main__":
    app = VisionaApp()
    app.run()