# pip install firebase-admin opencv-python pandas numpy sounddevice scipy

"""
Final Year Capstone Project: Multimodal Stress Detection Data Collector
This script records real-time sensor data from Firebase Realtime Database,
captures periodic webcam photos, and records continuous microphone audio.

Instructions to Run:
1. Install dependencies:
   pip install firebase-admin opencv-python pandas numpy sounddevice scipy
2. Place 'serviceAccountKey.json' in the same folder as this script.
3. Run the script:
   python collect_multimodal_data.py
4. Press CTRL+C to stop recording and save all session data.
"""

import os
import sys
import time
from datetime import datetime
import threading
import csv
import cv2
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# =====================================================================
# BACKGROUND MICROPHONE RECORDER
# =====================================================================
class AudioRecorder:
    def __init__(self, samplerate=44100, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.recording_data = []
        self.stream = None
        self.is_recording = False

    def callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.recording_data.append(indata.copy())

    def start(self):
        self.recording_data = []
        self.is_recording = True
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                callback=self.callback
            )
            self.stream.start()
            print("-> Microphone Recording Started...")
        except Exception as e:
            self.is_recording = False
            print(f"Warning: Microphone is unavailable ({e}). Audio will not be recorded.")

    def stop(self, filepath):
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error closing audio stream: {e}")
        if self.recording_data:
            try:
                audio_data = np.concatenate(self.recording_data, axis=0)
                wavfile.write(filepath, self.samplerate, audio_data)
                print(f"-> Audio Saved Successfully: {filepath}")
            except Exception as e:
                print(f"Error saving audio file: {e}")
        else:
            print("-> No audio data recorded.")

# =====================================================================
# BACKGROUND WEBCAM GRABBER (Avoids frame buffering delay)
# =====================================================================
class CameraManager:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.cap = None
        self.latest_frame = None
        self.ret = False
        self.running = False
        self.thread = None

    def start(self):
        try:
            # On Windows, cv2.CAP_DSHOW (DirectShow) is more reliable and faster to open.
            # We will search common indices (0, 1, 2) with both CAP_DSHOW and default backend.
            opened = False
            for backend in [cv2.CAP_DSHOW, None]:
                for idx in [self.device_index, 0, 1, 2]:
                    if backend is not None:
                        self.cap = cv2.VideoCapture(idx, backend)
                    else:
                        self.cap = cv2.VideoCapture(idx)
                    
                    if self.cap.isOpened():
                        # Try reading a test frame to ensure it actually works
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            self.device_index = idx
                            opened = True
                            break
                        else:
                            self.cap.release()
                if opened:
                    break

            if not opened:
                print("Warning: Laptop webcam is unavailable. Photos will not be captured.")
                return False

            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            print(f"-> Webcam Capture Ready (Device Index: {self.device_index})...")
            return True
        except Exception as e:
            print(f"Warning: Error initializing camera ({e}). Photos will not be captured.")
            return False


    def _update(self):
        while self.running:
            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    self.ret = True
                    self.latest_frame = frame
            time.sleep(0.03)  # Keep buffer empty (~30 FPS)

    def get_frame(self):
        return self.ret, self.latest_frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            print("-> Webcam Released...")

# =====================================================================
# MAIN APPLICATION
# =====================================================================
def main():
    print("=" * 60)
    print("      MULTIMODAL STRESS DETECTION SYSTEM - DATA COLLECTOR      ")
    print("=" * 60)

    # 1. USER INPUTS
    subject = input("Enter Subject Name: ").strip()
    if not subject:
        subject = "Unknown"
    
    condition = input("Enter Condition (Resting / Walking / Stress / Exercise): ").strip()
    if not condition:
        condition = "General"

    # Make safe directory names
    subject_safe = subject.replace(" ", "_")
    condition_safe = condition.replace(" ", "_")
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Session directories
    session_name = f"{subject_safe}_{condition_safe}_{timestamp_str}"
    session_dir = os.path.join("dataset", session_name)
    photos_dir = os.path.join(session_dir, "photos")

    try:
        os.makedirs(photos_dir, exist_ok=True)
        print(f"\nCreated session directory: {session_dir}")
    except Exception as e:
        print(f"Error creating session directories: {e}")
        sys.exit(1)

    # 2. FIREBASE SETUP
    print("\nConnecting to Firebase...")
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://wrist-band-6b8e4-default-rtdb.asia-southeast1.firebasedatabase.app/"
        })
        ref = db.reference("sensors")
        print("-> Firebase Connected Successfully.")
    except Exception as e:
        print(f"Error connecting to Firebase: {e}")
        print("Please verify 'serviceAccountKey.json' exists and database connection details are correct.")
        sys.exit(1)

    # 3. CSV FILE INITIALIZATION
    csv_file = os.path.join(session_dir, "sensor_data.csv")
    headers = [
        "Subject",
        "Condition",
        "Timestamp",
        "BPM",
        "GSR",
        "SpO2",
        "Temperature",
        "AccX",
        "AccY",
        "AccZ",
        "GyroX",
        "GyroY",
        "GyroZ",
        "PhotoFile"
    ]

    try:
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"-> Initialized CSV: {csv_file}")
    except Exception as e:
        print(f"Error initializing CSV: {e}")
        sys.exit(1)

    # 4. INITIALIZE PERIPHERALS
    audio_path = os.path.join(session_dir, "audio.wav")
    recorder = AudioRecorder()
    recorder.start()

    camera = CameraManager()
    if camera.start():
        print("Waiting 2 seconds for webcam to warm up...")
        time.sleep(2.0)


    print("\n" + "=" * 50)
    print("Recording Started. Collecting multimodal data every 10 seconds.")
    print("Press CTRL + C to Stop Recording")
    print("=" * 50 + "\n")

    photo_counter = 1
    next_time = time.time()

    # 5. DATA COLLECTION LOOP
    try:
        while True:
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # --- Capture Photo ---
            photo_file = "N/A"
            ret, frame = camera.get_frame()
            if ret and frame is not None:
                photo_file = f"photo_{photo_counter:03d}.jpg"
                photo_path = os.path.join(photos_dir, photo_file)
                try:
                    cv2.imwrite(photo_path, frame)
                    photo_counter += 1
                except Exception as e:
                    print(f"[{timestamp}] Error saving photo: {e}")
                    photo_file = "Error"
            else:
                print(f"[{timestamp}] Camera frame not available.")

            # --- Fetch Firebase Sensor Data ---
            bpm = "N/A"
            gsr = "N/A"
            spo2 = "N/A"
            temp = "N/A"
            ax, ay, az = "N/A", "N/A", "N/A"
            gx, gy, gz = "N/A", "N/A", "N/A"

            try:
                data = ref.get()
                if isinstance(data, dict):
                    bpm = data.get("bpm", "N/A")
                    gsr = data.get("gsr", "N/A")
                    spo2 = data.get("spo2", "N/A")
                    temp = data.get("temperature", "N/A")
                    
                    mpu = data.get("mpu", {})
                    if isinstance(mpu, dict):
                        ax = mpu.get("ax", "N/A")
                        ay = mpu.get("ay", "N/A")
                        az = mpu.get("az", "N/A")
                        gx = mpu.get("gx", "N/A")
                        gy = mpu.get("gy", "N/A")
                        gz = mpu.get("gz", "N/A")
                elif data is not None:
                    print(f"[{timestamp}] Unexpected Firebase structure format: {data}")
            except Exception as e:
                print(f"[{timestamp}] Error fetching Firebase data: {e}")

            # --- Log Data to CSV ---
            row_values = [
                subject,
                condition,
                timestamp,
                bpm,
                gsr,
                spo2,
                temp,
                ax,
                ay,
                az,
                gx,
                gy,
                gz,
                photo_file
            ]

            try:
                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row_values)
                
                print(f"[{timestamp}] Saved: BPM={bpm} | GSR={gsr} | Temp={temp} | Photo={photo_file}")
            except Exception as e:
                print(f"[{timestamp}] Error writing data to CSV: {e}")

            # Calculate sleep duration to stay exactly aligned with 10-second intervals
            next_time += 10
            sleep_duration = next_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                # If loop execution took longer than 10 seconds, reset next_time
                next_time = time.time()

    except KeyboardInterrupt:
        print("\n\n" + "=" * 50)
        print("Stopping recording session...")
        print("=" * 50)

    finally:
        # 6. GRACEFUL SHUTDOWN
        camera.stop()
        recorder.stop(audio_path)
        print("\nSession Saved Successfully!")
        print(f"Saved directory: {session_dir}")
        print("=" * 60)

if __name__ == "__main__":
    main()
