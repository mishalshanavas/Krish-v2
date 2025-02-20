from gtts import gTTS
from gtts.lang import tts_langs
import os
import sys

def text_to_speech(text, language='en', output_file='output.mp3'):
    """
    Convert text to speech using Google Text-to-Speech API
    
    Args:
        text (str): Text to convert to speech
        language (str): Language code (default: 'en')
        output_file (str): Output audio file name (default: 'output.mp3')
    """
    try:
        # Verify language is supported
        supported_langs = tts_langs()
        if language not in supported_langs:
            print(f"Error: Language code '{language}' is not supported")
            print("Supported languages:", ", ".join(supported_langs.keys()))
            return False
            
        # Create gTTS object
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to file
        tts.save(output_file)
        
        # Try to play the file (platform dependent)
        try:
            if sys.platform == 'darwin':  # macOS
                os.system(f'afplay {output_file}')
            elif sys.platform == 'win32':  # Windows
                os.system(f'start {output_file}')
            elif sys.platform == 'linux':  # Linux
                os.system(f'xdg-open {output_file}')
        except:
            print(f"Audio file saved as {output_file}")
            
        return True
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # Simple example
    text_to_speech("Hello! This is a test of text to speech conversion.")
    
    # Multiple language example
    text_to_speech("fuck u fuck u fuck u fuck u ", language='fr', output_file='french.mp3')
    
    # Longer text example
    long_text = """
    This is a longer piece of text that will be converted to speech.
    You can write multiple paragraphs and they will all be processed together.
    The gTTS library will handle the conversion appropriately.
    """
    text_to_speech(long_text, output_file='long_text.mp3')