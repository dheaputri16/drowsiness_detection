import mediapipe as mp
from scipy.spatial import distance
import numpy as np
import cv2
import pygame
import os
import struct
from collections import deque
import threading
import time

class DrowsinessDetector:
    def _init_(self,
                 eye_ar_thresh=0.40,  # Threshold for eye closure
                 mouth_ar_thresh=0.7,  # Threshold for yawning
                 head_movement_thresh=0.1,  # Normalized threshold for head movement
                 eye_weight=0.5,  # Weight for eye closure (scaled by duration)
                 yawn_weight=0.3,  # Weight for yawning
                 head_movement_weight=0.2,  # Weight for head movement
                 drowsiness_threshold=3.0,  # Threshold for alarm
                 max_drowsiness_level=5.0):  # Maximum drowsiness level
        # Detection thresholds
        self.eye_ar_thresh = eye_ar_thresh
        self.mouth_ar_thresh = mouth_ar_thresh
        self.head_movement_thresh = head_movement_thresh

        # Indicator weights
        self.eye_weight = eye_weight
        self.yawn_weight = yawn_weight
        self.head_movement_weight = head_movement_weight

        # Drowsiness parameters
        self.drowsiness_threshold = drowsiness_threshold
        self.max_drowsiness_level = max_drowsiness_level

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Landmark indices
        self.LEFT_EYE = [362, 385, 387, 373, 380, 374]
        self.RIGHT_EYE = [33, 160, 158, 153, 144, 145]
        self.MOUTH = [78, 308, 14, 13, 82, 312]
        self.FACE_BOUNDS = [10, 152]  # Forehead and chin for normalization

        # Tracking variables
        self.reset_counters()

        # Smoothing histories
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        self.head_movement_history = deque(maxlen=5)

        # Current indicators
        self.current_indicators = {
            'eyes_closed': 0.0,
            'yawning': 0.0,
            'head_movement': 0.0
        }

        # Drowsiness level
        self.drowsiness_level = 0.0

        # Eye closure duration tracking
        self.eye_closure_start_time = None
        self.eye_closure_duration_threshold = 2.0  # Seconds for prolonged closure

        # Detection log
        self.detection_log = []

        # Initialize audio
        self.init_audio()

        # Alarm status
        self.alarm_thread = None
        self.is_alarm_playing = False

        # Cooldown periods
        self.cooldowns = {
            'eyes_closed': 1.0,  # Allow frequent eye closure detections
            'yawning': 2.0,  # Yawns are less frequent
            'head_movement': 1.5  # Moderate frequency for head movement
        }
        self.last_detection_times = {
            'eyes_closed': 0,
            'yawning': 0,
            'head_movement': 0
        }

    def reset_counters(self):
        """Reset all counters and status variables"""
        self.alarm_on = False
        self.prev_landmarks = None
        self.drowsiness_level = 0.0
        self.eye_closure_start_time = None
        self.current_indicators = {
            'eyes_closed': 0.0,
            'yawning': 0.0,
            'head_movement': 0.0
        }
        self.detection_log = []
        self.last_detection_times = {
            'eyes_closed': 0,
            'yawning': 0,
            'head_movement': 0
        }
        print("[INFO] Drowsiness counters reset!")

    def init_audio(self):
        """Initialize pygame mixer for alarm"""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        except Exception as e:
            print(f"Warning: Cannot initialize audio: {e}")

    def eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR)"""
        if len(eye_points) < 6:
            return 0.0
        try:
            A = distance.euclidean(eye_points[1], eye_points[5])
            B = distance.euclidean(eye_points[2], eye_points[4])
            C = distance.euclidean(eye_points[0], eye_points[3])
            ear = (A + B) / (2.0 * C) if C > 0 else 0.0
            return ear
        except Exception as e:
            print(f"Error in EAR calculation: {e}")
            return 0.0

    def mouth_aspect_ratio(self, mouth_points):
        """Calculate Mouth Aspect Ratio (MAR)"""
        if len(mouth_points) < 6:
            return 0.0
        try:
            A = distance.euclidean(mouth_points[1], mouth_points[5])
            B = distance.euclidean(mouth_points[2], mouth_points[4])
            C = distance.euclidean(mouth_points[0], mouth_points[3])
            mar = (A + B) / (2.0 * C) if C > 0 else 0.0
            return mar
        except Exception as e:
            print(f"Error in MAR calculation: {e}")
            return 0.0

    def head_movement(self, landmarks, prev_landmarks):
        """Measure normalized head movement"""
        if prev_landmarks is None or landmarks is None:
            return 0.0
        try:
            face_size = distance.euclidean(
                (landmarks[10].x, landmarks[10].y),
                (landmarks[152].x, landmarks[152].y)
            )
            if face_size == 0:
                return 0.0
            nose_movement = distance.euclidean(
                (landmarks[1].x, landmarks[1].y),
                (prev_landmarks[1].x, prev_landmarks[1].y)
            )
            chin_movement = distance.euclidean(
                (landmarks[199].x, landmarks[199].y),
                (prev_landmarks[199].x, prev_landmarks[199].y)
            )
            movement = (nose_movement + chin_movement) / (2.0 * face_size)
            return movement
        except Exception as e:
            print(f"Error in head movement calculation: {e}")
            return 0.0

    def _play_simple_beep(self):
        """Play a simple beep"""
        try:
            duration = 2.0
            sample_rate = 44100
            bits = 16
            frequency = 440
            period = int(sample_rate / frequency)
            amplitude = 2**(bits-1) - 1
            buffer = bytearray()
            for i in range(int(sample_rate * duration)):
                value = amplitude * np.sin(2.0 * np.pi * float(i) / float(period))
                buffer.extend(struct.pack('h', int(value)))
            sound = pygame.mixer.Sound(buffer=buffer)
            sound.play()
            return True
        except Exception as e:
            print(f"Error playing beep: {e}")
            return False

    def _sound_alarm_thread(self):
        """Thread to play alarm"""
        self.is_alarm_playing = True
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        except:
            pass
        if os.path.exists("alarm.wav"):
            try:
                alarm_sound = pygame.mixer.Sound("alarm.wav")
                alarm_sound.play()
            except Exception as e:
                print(f"Error playing alarm.wav: {e}")
                self._play_simple_beep()
        else:
            self._play_simple_beep()
        time.sleep(3)
        self.is_alarm_playing = False

    def sound_alarm(self):
        """Play alarm in a separate thread"""
        if not self.is_alarm_playing:
            self.alarm_thread = threading.Thread(target=self._sound_alarm_thread)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()

    def get_landmark_points(self, face_landmarks, indices):
        """Extract x,y coordinates from landmarks"""
        points = []
        for idx in indices:
            if idx < len(face_landmarks.landmark):
                point = face_landmarks.landmark[idx]
                points.append((point.x, point.y))
        return points

    def add_drowsiness_indicator(self, indicator_type, value):
        """Add contribution to drowsiness level"""
        current_time = time.time()
        if current_time - self.last_detection_times[indicator_type] < self.cooldowns[indicator_type]:
            return
        if indicator_type == 'eyes_closed':
            contribution = value * self.eye_weight
        elif indicator_type == 'yawning':
            contribution = value * self.yawn_weight
        elif indicator_type == 'head_movement':
            contribution = value * self.head_movement_weight
        else:
            return
        self.last_detection_times[indicator_type] = current_time
        if contribution > 0:
            self.drowsiness_level += contribution
            self.drowsiness_level = min(self.drowsiness_level, self.max_drowsiness_level)
            self.current_indicators[indicator_type] = value
            log_entry = {
                'time': current_time,
                'type': indicator_type,
                'value': value,
                'contribution': contribution,
                'total': self.drowsiness_level
            }
            self.detection_log.append(log_entry)
            print(f"[DETECTION] {indicator_type}: +{contribution:.2f} → Total: {self.drowsiness_level:.2f}/{self.drowsiness_threshold}")

    def detect_drowsiness(self, frame):
        """Detect drowsiness from a frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            current_ear = 0.0
            current_mar = 0.0
            current_head_movement = 0.0
            status = "Normal"
            temp_indicators = {
                'eyes_closed': 0.0,
                'yawning': 0.0,
                'head_movement': 0.0
            }
            output_frame = frame.copy()
            if not results.multi_face_landmarks:
                cv2.putText(output_frame, "No Face Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                status = "No Face Detected"
                self.eye_closure_start_time = None  # Reset eye closure timer
            else:
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                left_eye_points = self.get_landmark_points(face_landmarks, self.LEFT_EYE)
                right_eye_points = self.get_landmark_points(face_landmarks, self.RIGHT_EYE)
                mouth_points = self.get_landmark_points(face_landmarks, self.MOUTH)
                if len(left_eye_points) >= 6:
                    left_eye_px = [(int(p[0] * w), int(p[1] * h)) for p in left_eye_points]
                    left_eye_hull = cv2.convexHull(np.array(left_eye_px, dtype=np.int32))
                    cv2.drawContours(output_frame, [left_eye_hull], -1, (0, 255, 0), 1)
                if len(right_eye_points) >= 6:
                    right_eye_px = [(int(p[0] * w), int(p[1] * h)) for p in right_eye_points]
                    right_eye_hull = cv2.convexHull(np.array(right_eye_px, dtype=np.int32))
                    cv2.drawContours(output_frame, [right_eye_hull], -1, (0, 255, 0), 1)
                if len(mouth_points) >= 6:
                    mouth_px = [(int(p[0] * w), int(p[1] * h)) for p in mouth_points]
                    mouth_hull = cv2.convexHull(np.array(mouth_px, dtype=np.int32))
                    cv2.drawContours(output_frame, [mouth_hull], -1, (0, 255, 0), 1)
                if len(left_eye_points) >= 6 and len(right_eye_points) >= 6:
                    left_ear = self.eye_aspect_ratio(left_eye_points)
                    right_ear = self.eye_aspect_ratio(right_eye_points)
                    ear = (left_ear + right_ear) / 2.0
                    self.ear_history.append(ear)
                    current_ear = sum(self.ear_history) / len(self.ear_history)
                    if current_ear < self.eye_ar_thresh:
                        if self.eye_closure_start_time is None:
                            self.eye_closure_start_time = time.time()
                        duration = time.time() - self.eye_closure_start_time
                        eyes_closed_value = min(1.0, duration / self.eye_closure_duration_threshold)
                        temp_indicators['eyes_closed'] = eyes_closed_value
                        if duration >= 0.5:  # Minimum duration to count
                            self.add_drowsiness_indicator('eyes_closed', eyes_closed_value)
                            cv2.putText(output_frame, f"EYES CLOSED ({duration:.1f}s)!", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            status = "Eyes Closed"
                    else:
                        self.eye_closure_start_time = None
                if len(mouth_points) >= 6:
                    mar = self.mouth_aspect_ratio(mouth_points)
                    self.mar_history.append(mar)
                    current_mar = sum(self.mar_history) / len(self.mar_history)
                    if current_mar > self.mouth_ar_thresh:
                        yawn_value = min(1.0, (current_mar - self.mouth_ar_thresh) / self.mouth_ar_thresh)
                        temp_indicators['yawning'] = yawn_value
                        self.add_drowsiness_indicator('yawning', yawn_value)
                        cv2.putText(output_frame, "YAWNING DETECTED!", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        status = "Yawning"
                movement = self.head_movement(face_landmarks.landmark, self.prev_landmarks)
                self.head_movement_history.append(movement)
                current_head_movement = sum(self.head_movement_history) / len(self.head_movement_history)
                if current_head_movement > self.head_movement_thresh:
                    head_value = min(1.0, (current_head_movement - self.head_movement_thresh) / self.head_movement_thresh)
                    temp_indicators['head_movement'] = head_value
                    self.add_drowsiness_indicator('head_movement', head_value)
                    cv2.putText(output_frame, "EXCESSIVE HEAD MOVEMENT!", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    status = "Head Movement"
                self.prev_landmarks = face_landmarks.landmark
                cv2.putText(output_frame, f"EAR: {current_ear:.2f}/{self.eye_ar_thresh:.2f}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(output_frame, f"MAR: {current_mar:.2f}/{self.mouth_ar_thresh:.2f}", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(output_frame, f"Head: {current_head_movement:.4f}/{self.head_movement_thresh:.4f}", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(output_frame, f"Drowsiness: {self.drowsiness_level:.2f}/{self.drowsiness_threshold:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            bar_length = int((self.drowsiness_level / self.max_drowsiness_level) * 200)
            cv2.rectangle(output_frame, (10, 350), (10 + bar_length, 370), (0, 0, 255), -1)
            cv2.rectangle(output_frame, (10, 350), (210, 370), (255, 255, 255), 2)
            is_drowsy = self.drowsiness_level >= self.drowsiness_threshold
            if is_drowsy:
                if not self.alarm_on:
                    self.alarm_on = True
                    self.sound_alarm()
                cv2.putText(output_frame, "DROWSY! TAKE A BREAK", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(output_frame, (0, 0), (output_frame.shape[1], output_frame.shape[0]), (0, 0, 255), 5)
                status = "DROWSY - ALERT!"
            else:
                self.alarm_on = False
            return {
                'frame': output_frame,
                'drowsiness_level': self.drowsiness_level,
                'is_drowsy': is_drowsy,
                'status': status,
                'indicators': {
                    'ear': current_ear,
                    'mar': current_mar,
                    'head_movement': current_head_movement
                }
            }
        except Exception as e:
            print(f"[ERROR] Error in detect_drowsiness: {e}")
            return {
                'frame': frame,
                'drowsiness_level': self.drowsiness_level,
                'is_drowsy': False,
                'status': f"Error: {str(e)}",
                'indicators': {
                    'ear': 0,
                    'mar': 0,
                    'head_movement': 0
                }
            }

    def get_status(self):
        """Get current status"""
        return {
            'drowsiness_level': self.drowsiness_level,
            'is_drowsy': self.drowsiness_level >= self.drowsiness_threshold,
            'alarm_on': self.alarm_on,
            'indicators': self.current_indicators,
            'detection_history': self.detection_log
        }

    def set_thresholds(self, **kwargs):
        """Dynamically set thresholds"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Set {key} = {value}")

    def cleanup(self):
        """Clean up resources"""
        try:
            pygame.mixer.quit()
        except:
            pass
        self.face_mesh.close()

if _name_ == "_main_":
    print("[DEBUG] DrowsinessDetector instance created")
    detector = DrowsinessDetector(
        eye_ar_thresh=0.40,
        mouth_ar_thresh=0.7,
        head_movement_thresh=0.1,
        eye_weight=0.5,
        yawn_weight=0.3,
        head_movement_weight=0.2,
        drowsiness_threshold=3.0,
        max_drowsiness_level=5.0
    )
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting drowsiness detection...")
    print("Press 'q' to quit, 'r' to reset counters")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame from camera.")
            break
        result = detector.detect_drowsiness(frame)
        cv2.imshow("Drowsiness Detection", result['frame'])
        if result['is_drowsy']:
            print(f"DROWSY DETECTED! Level: {result['drowsiness_level']:.2f}, Indicators: {result['indicators']}")
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            detector.reset_counters()
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
    print("[INFO] Program terminated.")