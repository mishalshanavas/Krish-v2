import os
import struct
import tempfile
import wave
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import contextmanager
from typing import Optional, Generator
from functools import lru_cache
import threading

import pvporcupine
import pyaudio
import google.generativeai as genai
import speech_recognition as sr
import gtts
import pygame
import numpy as np
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AudioConfig:
    """Audio configuration constants."""
    SAMPLE_RATE: int = 44100
    CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2
    DETECTION_FREQ: float = 1000.0
    DETECTION_DURATION: float = 0.1
    TIMEOUT_FREQ: float = 800.0
    TIMEOUT_DURATION: float = 0.15
    LISTEN_TIMEOUT: int = 8

@dataclass(frozen=True)
class WakeWord:
    """Wake word configuration."""
    phrase: str
    model_path: str
    sensitivity: float
    language: str

class Language(Enum):
    """Supported languages."""
    ENGLISH = auto()
    MALAYALAM = auto()
    
    @property
    def speech_code(self) -> str:
        return "en-US" if self == Language.ENGLISH else "ml-IN"
    
    @property
    def tts_config(self) -> tuple[str, str]:
        return ("en", "co.in") if self == Language.ENGLISH else ("ml", "com")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLE STYLING
# ═══════════════════════════════════════════════════════════════════════════════

class Console:
    """Styled console output."""
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    
    @classmethod
    def banner(cls) -> None:
        print(f"""{cls.PURPLE}{cls.BOLD}
    ╔═══════════════════════════════════════════════════════════╗
    ║   �DiNO🦖                                                 ║
    ║   ██╗  ██╗██████╗ ██╗███████╗██╗  ██╗███╗   ██╗ █████╗   ║
    ║   ██║ ██╔╝██╔══██╗██║██╔════╝██║  ██║████╗  ██║██╔══██╗  ║
    ║   █████╔╝ ██████╔╝██║███████╗███████║██╔██╗ ██║███████║  ║
    ║   ██╔═██╗ ██╔══██╗██║╚════██║██╔══██║██║╚██╗██║██╔══██║  ║
    ║   ██║  ██╗██║  ██║██║███████║██║  ██║██║ ╚████║██║  ██║  ║
    ║   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝  ║
    ║                    🎤 Voice Assistant                     ║
    ╚═══════════════════════════════════════════════════════════╝{cls.RESET}
        """)
    
    @classmethod
    def status(cls, icon: str, msg: str, color: str = "") -> None:
        print(f"{color}{cls.BOLD}  {icon}  {msg}{cls.RESET}")
    
    @classmethod
    def listening(cls) -> None:
        cls.status("👂", "Listening...", cls.CYAN)
    
    @classmethod
    def wake_detected(cls, word: str) -> None:
        cls.status("✨", f"Wake word detected: {word}", cls.GREEN)
    
    @classmethod
    def user_said(cls, text: str) -> None:
        cls.status("🗣️ ", f"You: {text}", cls.YELLOW)
    
    @classmethod
    def assistant_says(cls, text: str) -> None:
        cls.status("🤖", f"Krishna: {text}", cls.PURPLE)
    
    @classmethod
    def error(cls, msg: str) -> None:
        cls.status("❌", msg, cls.RED)
    
    @classmethod
    def info(cls, msg: str) -> None:
        cls.status("ℹ️ ", msg, cls.DIM)

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceAssistant:
    """Krishna AI Voice Assistant - Efficient & Modern Implementation."""
    
    WAKE_WORDS = [
        WakeWord("Hey Krishna", "Hey-Krishna_en_linux_v3_0_0.ppn", 0.3, "eng"),
        WakeWord("Hello Krishna", "Hello-krishna_en_linux_v3_0_0.ppn", 0.4, "mal"),
    ]
    
    SYSTEM_PROMPT = """You are Krishna AI assistant, not Gemini anymore. Keep your responses:
    1. Brief (2-3 sentences maximum)
    2. Conversational and natural for students under 18
    3. Avoid special characters, symbols, or formatting
    4. Use simple, clear language
    5. Round numbers for easier speech
    6. Give responses in an Indian, Kerala context if relevant
    7. Answer educational and general knowledge questions"""

    def __init__(self):
        load_dotenv()
        self._validate_env()
        self._audio_config = AudioConfig()
        self._temp_files: list[str] = []
        self._lock = threading.Lock()
        
        # Initialize all components
        self._init_gemini()
        self._init_audio()
        self._init_speech()
        self._init_sounds()

    def _validate_env(self) -> None:
        """Validate required environment variables."""
        self._gemini_key = os.getenv("GEMINI_API_KEY")
        self._picovoice_key = os.getenv("PICOVOICE_ACCESS_KEY")
        
        if not all([self._gemini_key, self._picovoice_key]):
            raise EnvironmentError("❌ Missing API keys! Check your .env file.")

    def _init_gemini(self) -> None:
        """Initialize Gemini AI with conversation history."""
        genai.configure(api_key=self._gemini_key)
        self._model = genai.GenerativeModel("gemini-2.0-flash-lite-preview-02-05")
        self._chat = self._model.start_chat(history=[
            {"role": "user", "parts": [self.SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Understood! I'm Krishna AI, ready to help students."]}
        ])

    def _init_audio(self) -> None:
        """Initialize Porcupine and audio stream."""
        self._porcupine = pvporcupine.create(
            access_key=self._picovoice_key,
            keyword_paths=[w.model_path for w in self.WAKE_WORDS],
            sensitivities=[w.sensitivity for w in self.WAKE_WORDS]
        )
        
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._porcupine.sample_rate,
            input=True,
            frames_per_buffer=self._porcupine.frame_length
        )

    def _init_speech(self) -> None:
        """Initialize speech recognition."""
        self._recognizer = sr.Recognizer()
        self._mic = sr.Microphone()
        # Pre-calibrate for faster response
        with self._mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=0.5)

    def _init_sounds(self) -> None:
        """Generate feedback sounds."""
        self._detection_beep = self._generate_beep(
            self._audio_config.DETECTION_FREQ,
            self._audio_config.DETECTION_DURATION
        )
        self._timeout_beep = self._generate_beep(
            self._audio_config.TIMEOUT_FREQ,
            self._audio_config.TIMEOUT_DURATION
        )

    @lru_cache(maxsize=4)
    def _generate_beep(self, frequency: float, duration: float) -> str:
        """Generate and cache a beep sound file."""
        cfg = self._audio_config
        t = np.linspace(0, duration, int(cfg.SAMPLE_RATE * duration))
        # Apply envelope for smoother sound
        envelope = np.sin(np.pi * t / duration)
        samples = (32767 * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self._temp_files.append(temp_file.name)
        
        with wave.open(temp_file.name, 'w') as wf:
            wf.setnchannels(cfg.CHANNELS)
            wf.setsampwidth(cfg.SAMPLE_WIDTH)
            wf.setframerate(cfg.SAMPLE_RATE)
            wf.writeframes(samples.tobytes())
        
        return temp_file.name

    @contextmanager
    def _audio_context(self) -> Generator[None, None, None]:
        """Thread-safe audio playback context."""
        with self._lock:
            pygame.mixer.init()
            try:
                yield
            finally:
                pygame.mixer.quit()

    def _play_sound(self, filepath: str) -> None:
        """Play a sound file."""
        with self._audio_context():
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

    def _beep(self, beep_type: str = "detect") -> None:
        """Play feedback beep."""
        sound = self._detection_beep if beep_type == "detect" else self._timeout_beep
        self._play_sound(sound)
        if beep_type == "timeout":
            pygame.time.wait(150)
            self._play_sound(sound)

    def speak(self, text: str, lang: Language = Language.ENGLISH) -> None:
        """Convert text to speech with optimized playback."""
        Console.assistant_says(text)
        
        try:
            lang_code, tld = lang.tts_config
            tts = gtts.gTTS(text=text, lang=lang_code, tld=tld, slow=False)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                tts.save(f.name)
                self._play_sound(f.name)
                os.unlink(f.name)
                
        except Exception as e:
            Console.error(f"TTS failed: {e}")

    def think(self, query: str) -> str:
        """Get AI response using conversation history."""
        try:
            response = self._chat.send_message(
                f"Answer briefly in 2-3 sentences for voice: {query}"
            )
            return response.text.strip()
        except Exception as e:
            Console.error(f"AI error: {e}")
            return "Sorry, I couldn't process that right now."

    def listen(self, lang: Language = Language.ENGLISH) -> Optional[str]:
        """Listen for voice command with timeout."""
        Console.listening()
        
        try:
            with self._mic as source:
                audio = self._recognizer.listen(
                    source, 
                    timeout=self._audio_config.LISTEN_TIMEOUT
                )
                command = self._recognizer.recognize_google(
                    audio, 
                    language=lang.speech_code
                )
                Console.user_said(command)
                return command
                
        except sr.WaitTimeoutError:
            self._beep("timeout")
            Console.info("No speech detected")
            return None
        except sr.UnknownValueError:
            self._beep("timeout")
            self.speak("Sorry, I didn't catch that.")
            return None
        except sr.RequestError:
            self._beep("timeout")
            self.speak("Having trouble connecting to speech services.")
            return None

    def run(self) -> None:
        """Main loop with improved error handling and shutdown."""
        self.logger.info(f"Listening for wake words: {[w[0] for w in self.WAKE_WORDS]}")
        
        try:
            while True:
                pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm_unpacked)
                if keyword_index >= 0:
                    detected_wake_word = self.WAKE_WORDS[keyword_index][0]
                    self.logger.info(f"Wake word detected: {detected_wake_word}")
                    
                    # Play detection beep
                    self._play_detection_beep()
                    
                    lang = "eng" if detected_wake_word == "Hey Krishna" else "mal"
                    self.logger.info(f"Listening for command... ({lang})")
                    
                    user_query = self.listen_for_command(lang)
                    if user_query:
                        response = self.chat_with_gemini(user_query)
                        self.talk(response, "english" if lang == "eng" else "malayalam")
                    
        except KeyboardInterrupt:
            self.logger.info("Shutting down voice assistant...")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.stream.close()
            self.pa.terminate()
            self.porcupine.delete()
            
            # Clean up temporary sound files
            os.unlink(self.detection_beep.name)
            os.unlink(self.timeout_beep.name)
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

def main():
    try:
        assistant = VoiceAssistant()
        assistant.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()