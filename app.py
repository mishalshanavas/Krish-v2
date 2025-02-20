import os
import pvporcupine
import pyaudio
import struct
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import tempfile
from dotenv import load_dotenv
from playsound import playsound

# Load API keys from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

# Wake word paths
WAKE_WORDS = [
    ("Hey Krishna", "Hey-Krishna_en_linux_v3_0_0.ppn"),
    ("Hello Krishna", "Hello-krishna_en_linux_v3_0_0.ppn")
]

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
chat = genai.GenerativeModel("gemini-pro")

# Initialize chat with context
SYSTEM_PROMPT = """You are Krishna AI assistant, not Gemini anymore. Keep your responses:
1. Brief (2-3 sentences maximum)
2. Conversational and natural for students under 18
3. Avoid special characters, symbols, or formatting
4. Use simple, clear language
5. Round numbers for easier speech
6. Give responses in an Indian, Kerala context if relevant
7. Answer educational and general knowledge questions, not smart home or alarm-related queries.
8. if a question is asked in malayalam 
Remember, you're speaking, not writing."""

chat_history = [
    {"role": "user", "parts": [SYSTEM_PROMPT]},
    {"role": "model", "parts": ["Understood, I'll keep responses brief and speech-friendly."]}
]

# Initialize Porcupine with custom wake words
try:
    porcupine = pvporcupine.create(
        access_key=PICOVOICE_ACCESS_KEY,
        keyword_paths=[w[1] for w in WAKE_WORDS],
        sensitivities=[0.3, 0.4]
    )
except Exception as e:
    print(f"Error initializing Porcupine: {str(e)}")
    print("Make sure your custom wake word file (.ppn) exists and the path is correct.")
    exit(1)

# Set up microphone
pa = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=porcupine.sample_rate,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

# Set up speech recognition
recognizer = sr.Recognizer()
mic = sr.Microphone()

def talk(text):
    """Convert text to speech using gTTS and play the audio."""
    print(f"Assistant: {text}")
    tts = gTTS(text=text, lang="en", tld="co.in")
    
    try:
        temp_file = os.path.join(tempfile.gettempdir(), "temp_speech.mp3")
        tts.save(temp_file)
        playsound(temp_file)
        try:
            os.remove(temp_file)
        except:
            pass
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")

def chat_with_gemini(query):
    """Send query to Gemini AI and get a voice-optimized response."""
    try:
        full_prompt = f"Remember you're a voice assistant. Answer this briefly in 2-3 sentences using simple speech: {query}"
        response = chat.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return "Sorry, I couldn't process that request right now."

import speech_recognition as sr
from playsound import playsound

recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_for_command(lang):
    """Listen for a spoken command after wake word is detected."""
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for your command...")
        try:
            audio = recognizer.listen(source, timeout=8)
            
            # Detect language
            if lang == "mal":
                command = recognizer.recognize_google(audio, language="ml-IN")
            else:
                command = recognizer.recognize_google(audio)

            print(f"You: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            try:
                playsound("notdetected.mp3")
            except Exception as e:
                print(f"Error playing detection sound: {str(e)}")
            return None
        except sr.RequestError:
            print("Speech recognition service is unavailable.")
            return None
    
def main():
    print("Listening for wake words:", [w[0] for w in WAKE_WORDS])

    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)

            keyword_index = porcupine.process(pcm_unpacked)
            if keyword_index >= 0:
                detected_wake_word = WAKE_WORDS[keyword_index][0]  # Get the wake word name
                print(f"Wake word detected: {detected_wake_word}")

                try:
                    playsound("detected.mp3")
                except Exception as e:
                    print(f"Error playing detection sound: {str(e)}")
                
                if detected_wake_word == "Hey Krishna":
                    print("Listening for your command...(English)") 
                    user_query = listen_for_command(lang="eng")
                elif detected_wake_word == "Hello Krishna":
                    print("Listening for your command...(Malayalam)")
                    user_query = listen_for_command(lang="mal")
                if user_query:
                    response = chat_with_gemini(user_query)
                    talk(response)
                    
    except KeyboardInterrupt:
        print("\nStopping the voice assistant...")
    finally:
        stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()
