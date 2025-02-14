import speech_recognition as sr
import pyttsx3
import webbrowser
import pyautogui
import os
import cv2
import mediapipe as mp
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import psutil
from datetime import datetime
import screen_brightness_control as sbc
import platform
import subprocess

class AIAssistant:
    def __init__(self):
        # Initialize main window
        self.window = tk.Tk()
        self.window.title("AI Assistant - Mouth Control")
        self.window.geometry("1280x720")
        self.window.configure(bg='#1E1E1E')

        # Initialize state variables
        self.is_listening = False
        self.shorts_mode = False
        self.camera_active = True
        self.last_mouth_action_time = time.time()
        self.mouth_cooldown = 0.3
        self.neutral_mouth_position = None
        self.brightness_level = 50

        # Initialize speech components
        self.recognizer = sr.Recognizer()
        self.speaker = pyttsx3.init()

        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create GUI elements
        self.create_gui()

        # Start camera
        self.camera = cv2.VideoCapture(0)

    def create_gui(self):
        main_container = tk.Frame(self.window, bg='#1E1E1E')
        main_container.pack(expand=True, fill='both', padx=20, pady=20)

        left_panel = tk.Frame(main_container, bg='#2D2D2D')
        left_panel.pack(side='left', fill='both', expand=True, padx=10)
        self.camera_label = tk.Label(left_panel, bg='#2D2D2D')
        self.camera_label.pack(pady=10)

        status_frame = tk.Frame(left_panel, bg='#2D2D2D')
        status_frame.pack(fill='x', pady=10)

        self.shorts_status = tk.Label(
            status_frame,
            text="Shorts Mode: OFF",
            font=("Arial", 12),
            bg='#2D2D2D',
            fg='#FF4444'
        )
        self.shorts_status.pack(side='left', padx=10)

        self.brightness_label = tk.Label(
            status_frame,
            text="Brightness: 50%",
            font=("Arial", 12),
            bg='#2D2D2D',
            fg='#00FF00'
        )
        self.brightness_label.pack(side='right', padx=10)

        right_panel = tk.Frame(main_container, bg='#2D2D2D')
        right_panel.pack(side='right', fill='both', expand=True, padx=10)

        self.text_area = tk.Text(
            right_panel,
            height=20,
            width=50,
            bg='#363636',
            fg='#FFFFFF',
            font=("Arial", 12)
        )
        self.text_area.pack(pady=20)

        instructions = """
        Voice Commands:
        • "Open YouTube"
        • "Turn on/off YouTube shorts"
        • "Open PW video player" (append "chrome" to launch as a Chrome App)
        • "Exit"
        
        Mouth Controls:
        In Shorts Mode:
          - Move mouth left/right → Previous/Next Short
          - Move mouth up/down → Scroll Up/Down

        When Shorts Mode is OFF:
          - Move mouth up/down → Increase/Decrease Brightness
        
        Current Time: {}
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.instructions_label = tk.Label(
            right_panel,
            text=instructions,
            justify='left',
            bg='#2D2D2D',
            fg='#FFFFFF',
            font=("Arial", 12)
        )
        self.instructions_label.pack(pady=10)

    def get_mouth_position(self, face_landmarks):
        left_mouth = face_landmarks.landmark[78]
        right_mouth = face_landmarks.landmark[308]
        top_mouth = face_landmarks.landmark[13]
        bottom_mouth = face_landmarks.landmark[14]
        center_x = (left_mouth.x + right_mouth.x) / 2
        center_y = (top_mouth.y + bottom_mouth.y) / 2
        return center_x, center_y

    def process_mouth_movement(self, face_landmarks):
        current_time = time.time()
        if current_time - self.last_mouth_action_time < self.mouth_cooldown:
            return

        current_x, current_y = self.get_mouth_position(face_landmarks)
        if self.neutral_mouth_position is None:
            self.neutral_mouth_position = (current_x, current_y)
            return

        move_x = current_x - self.neutral_mouth_position[0]
        move_y = current_y - self.neutral_mouth_position[1]
        threshold = 0.03

        if self.shorts_mode:
            if abs(move_x) > threshold:
                if move_x > 0:
                    pyautogui.press('right')
                    self.log_action("Mouth: Next Short")
                else:
                    pyautogui.press('left')
                    self.log_action("Mouth: Previous Short")
                self.last_mouth_action_time = current_time
            elif abs(move_y) > threshold:
                if move_y < 0:
                    pyautogui.press('up')
                    self.log_action("Mouth: Scrolling Up")
                else:
                    pyautogui.press('down')
                    self.log_action("Mouth: Scrolling Down")
                self.last_mouth_action_time = current_time
        else:
            if abs(move_y) > threshold:
                try:
                    current_brightness = sbc.get_brightness()[0]
                    change = 10 if move_y < 0 else -10
                    new_brightness = max(0, min(100, current_brightness + change))
                    sbc.set_brightness(new_brightness)
                    self.brightness_label.config(text=f"Brightness: {new_brightness}%")
                    self.log_action(f"Brightness adjusted to {new_brightness}%")
                    self.last_mouth_action_time = current_time
                except Exception as e:
                    self.log_action(f"Brightness control error: {e}")

        self.neutral_mouth_position = (
            0.95 * self.neutral_mouth_position[0] + 0.05 * current_x,
            0.95 * self.neutral_mouth_position[1] + 0.05 * current_y
        )

    def update_camera(self):
        if self.camera_active and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = self.face_mesh.process(rgb_frame)
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                                color=(0, 255, 0),
                                thickness=1,
                                circle_radius=1
                            )
                        )
                        self.process_mouth_movement(face_landmarks)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = ImageTk.PhotoImage(image=img)
                self.camera_label.configure(image=img)
                self.camera_label.image = img
            self.window.after(10, self.update_camera)

    def log_action(self, text):
        self.text_area.insert('end', f"{text}\n")
        self.text_area.see('end')

    def speak(self, text):
        self.log_action(f"Assistant: {text}")
        self.speaker.say(text)
        self.speaker.runAndWait()

    def open_pw_video_player_in_native(self):
        sys_platform = platform.system()
        if sys_platform == "Windows":
            video_player_path = r"C:\Program Files\PWVideoPlayer\pwplayer.exe"
            try:
                os.startfile(video_player_path)
                self.speak("Opening PW Video Player")
            except Exception as e:
                self.speak("Failed to open PW Video Player: " + str(e))
        elif sys_platform == "Darwin":
            video_player_path = "/Applications/PWVideoPlayer.app"
            try:
                os.system(f"open '{video_player_path}'")
                self.speak("Opening PW Video Player")
            except Exception as e:
                self.speak("Failed to open PW Video Player: " + str(e))
        else:
            video_player_path = "/usr/bin/pwvideoplayer"
            try:
                os.system(f"xdg-open '{video_player_path}'")
                self.speak("Opening PW Video Player")
            except Exception as e:
                self.speak("Failed to open PW Video Player: " + str(e))

    def open_pw_video_player_in_chrome(self):
        sys_platform = platform.system()
        if sys_platform == "Windows":
            # Use the shortcut from the specified location.
            chrome_app_path = r"C:\Users\Narvat\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Chrome Apps\PW Video Player.lnk"
            try:
                os.startfile(chrome_app_path)
                self.speak("Opening PW Video Player in Chrome App")
            except Exception as e:
                self.speak("Failed to open PW Video Player in Chrome App: " + str(e))
        elif sys_platform == "Darwin":
            chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            video_url = "file:///Applications/PWVideoPlayer/index.html"
            try:
                subprocess.Popen([chrome_path, "--app=" + video_url])
                self.speak("Opening PW Video Player in Chrome App")
            except Exception as e:
                self.speak("Failed to open in Chrome App: " + str(e))
        else:
            chrome_path = "google-chrome"
            video_url = "file:///usr/local/PWVideoPlayer/index.html"
            try:
                subprocess.Popen([chrome_path, "--app=" + video_url])
                self.speak("Opening PW Video Player in Chrome App")
            except Exception as e:
                self.speak("Failed to open in Chrome App: " + str(e))

    def listen_loop(self):
        while self.is_listening:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source)
                    text = self.recognizer.recognize_google(audio).lower()
                    self.log_action(f"You: {text}")
                    self.process_command(text)
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print("Error: ", e)

    def process_command(self, command):
        if "open youtube" in command:
            webbrowser.open("https://www.youtube.com")
            self.speak("Opening YouTube")
        elif "youtube shorts" in command or "short" in command:
            if "turn on" in command or "open" in command:
                self.shorts_mode = True
                webbrowser.open("https://www.youtube.com/shorts")
                time.sleep(2)
                pyautogui.press('f')
                self.shorts_status.config(text="Shorts Mode: ON", fg='#00FF00')
                self.speak("YouTube Shorts mode activated.")
            elif "turn off" in command or "close" in command:
                self.shorts_mode = False
                pyautogui.press('f')
                time.sleep(1)
                webbrowser.open("https://www.youtube.com")
                self.shorts_status.config(text="Shorts Mode: OFF", fg='#FF4444')
                self.speak("YouTube Shorts mode deactivated.")
        elif "open pw video player" in command:
            if "chrome" in command:
                self.open_pw_video_player_in_chrome()
            else:
                self.open_pw_video_player_in_native()
        elif "exit" in command:
            self.cleanup()
            self.window.quit()

    def start(self):
        self.is_listening = True
        threading.Thread(target=self.listen_loop, daemon=True).start()
        self.update_camera()
        self.window.mainloop()

    def cleanup(self):
        self.is_listening = False
        self.camera_active = False
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    assistant = AIAssistant()
    try:
        assistant.start()
    except Exception as e:
        print("Error: ", e)
    finally:
        assistant.cleanup()
