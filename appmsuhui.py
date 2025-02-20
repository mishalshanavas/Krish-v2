import os
import pvporcupine
import pyaudio
import struct
import google.generativeai as genai
import speech_recognition as sr
import gtts
import pygame
import tempfile
import wave
import numpy as np
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Optional, Tuple, List
import logging

class VoiceAssistant:
    def __init__(self):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        load_dotenv()
        self._initialize_api_keys()
        
        # Configure wake words
        self.WAKE_WORDS = [
            ("Hey Krishna", "Hey-Krishna_en_linux_v3_0_0.ppn"),
            ("Hello Krishna", "Hello-krishna_en_linux_v3_0_0.ppn")
        ]
        
        # Initialize components
        self._setup_gemini()
        self._setup_audio()
        self._setup_speech_recognition()
        self._generate_beep_sounds()

    def _generate_beep_sounds(self) -> None:
        """Generate beep sound files for feedback."""
        def generate_beep(frequency: float, duration: float, filename: str) -> None:
            # Generate a single beep
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration))
            samples = (32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
            
            # Save as WAV file
            with wave.open(filename, 'w') as wave_file:
                wave_file.setnchannels(1)
                wave_file.setsampwidth(2)
                wave_file.setframerate(sample_rate)
                wave_file.writeframes(samples.tobytes())

        # Generate detection beep (higher frequency, shorter duration)
        self.detection_beep = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        generate_beep(1000, 0.1, self.detection_beep.name)

        # Generate timeout beep (lower frequency, longer duration)
        self.timeout_beep = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        generate_beep(800, 0.15, self.timeout_beep.name)

    def _play_sound(self, sound_file: str) -> None:
        """Play a sound file using pygame."""
        with self._audio_playback_context():
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

    def _play_detection_beep(self) -> None:
        """Play detection beep sound."""
        self._play_sound(self.detection_beep.name)

    def _play_timeout_beeps(self) -> None:
        """Play two timeout beep sounds."""
        for _ in range(2):
            self._play_sound(self.timeout_beep.name)
            pygame.time.wait(200)  # Wait 200ms between beeps

    def _initialize_api_keys(self) -> None:
        """Initialize API keys with error handling."""
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.picovoice_key = os.getenv("PICOVOICE_ACCESS_KEY")
        
        if not self.gemini_key or not self.picovoice_key:
            raise EnvironmentError("Missing required API keys in .env file")

    def _setup_gemini(self) -> None:
        """Configure Gemini AI with system prompt."""
        genai.configure(api_key=self.gemini_key)
        self.chat = genai.GenerativeModel("gemini-2.0-flash-lite-preview-02-05")
        
        SYSTEM_PROMPT = """You are Krishna AI assistant, not Gemini anymore. Keep your responses:
        1. Brief (2-3 sentences maximum)
        2. Conversational and natural for students under 18
        3. Avoid special characters, symbols, or formatting
        4. Use simple, clear language
        5. Round numbers for easier speech
        6. Give responses in an Indian, Kerala context if relevant
        7. Answer educational and general knowledge questions
        """
        
        self.chat_history = [
            {"role": "user", "parts": [SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Understood, I'll keep responses brief and speech-friendly."]}
        ]

    def _setup_audio(self) -> None:
        """Initialize Porcupine and PyAudio with error handling."""
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.picovoice_key,
                keyword_paths=[w[1] for w in self.WAKE_WORDS],
                sensitivities=[0.3, 0.4]
            )
            
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.porcupine.sample_rate,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
        except Exception as e:
            self.logger.error(f"Audio setup failed: {str(e)}")
            raise

    def _setup_speech_recognition(self) -> None:
        """Initialize speech recognition components."""
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    @contextmanager
    def _audio_playback_context(self):
        """Context manager for pygame audio playback."""
        pygame.mixer.init()
        try:
            yield
        finally:
            pygame.mixer.quit()

    def talk(self, text: str, language: str = "english") -> None:
        """Convert text to speech with improved error handling and resource management."""
        self.logger.info(f"Assistant: {text}")
        
        try:
            # Configure language settings
            if language == "english":
                lang_code = 'en'
                tld = 'co.in'
            else:  # Malayalam
                lang_code = 'ml'
                tld = 'com'  # Fixed: Use 'com' instead of None for Malayalam
            
            tts = gtts.gTTS(text=text, lang=lang_code, tld=tld, slow=False)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                tts.save(temp_file.name)
                
                with self._audio_playback_context():
                    pygame.mixer.music.load(temp_file.name)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                
                os.unlink(temp_file.name)
                
        except Exception as e:
            self.logger.error(f"TTS Error ({language}): {str(e)}")
            if language == "english":
                # Only attempt fallback for English to avoid recursion
                self.talk("Sorry, I'm having trouble speaking right now.", "english")

    def chat_with_gemini(self, query: str) -> str:
        """Get AI response with improved error handling."""
        try:
            full_prompt = f"Remember you're a voice assistant. Answer this briefly in 2-3 sentences using simple speech: {query}"
            response = self.chat.generate_content(full_prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini API error: {str(e)}")
            return "Sorry, I couldn't process that request right now."

    def listen_for_command(self, lang: str) -> Optional[str]:
        """Listen for voice command with timeout and error handling."""
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.logger.info("Listening for your command...")
            
            try:
                audio = self.recognizer.listen(source, timeout=8)
                language_code = "ml-IN" if lang == "mal" else "en-US"
                command = self.recognizer.recognize_google(audio, language=language_code)
                self.logger.info(f"User: {command}")
                return command
                
            except sr.UnknownValueError:
                self._play_timeout_beeps()
                self.talk("Sorry, I didn't catch that.")
                return None
            except sr.RequestError:
                self._play_timeout_beeps()
                self.talk("Sorry, I'm having trouble understanding you right now.")
                return None
            except Exception as e:
                self.logger.error(f"Listening error: {str(e)}")
                self._play_timeout_beeps()
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