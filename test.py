import os
import gtts
import pygame
import tempfile

def speak_response(response, language="english"):
    # Initialize pygame mixer
    pygame.mixer.init()
    
    try:
        # Configure TTS based on language
        if language == "english":
            tts = gtts.gTTS(text=response, lang='en', tld='co.in')
        else:
            tts = gtts.gTTS(text=response, lang='ml', slow=False)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            # Save audio to temp file
            tts.save(temp_file.name)
            
            try:
                # Play the audio
                pygame.mixer.music.load(temp_file.name)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
            finally:
                # Cleanup
                pygame.mixer.music.unload()
                pygame.mixer.quit()
                os.unlink(temp_file.name)
                
    except Exception as e:
        print(f"TTS Error ({language}): {str(e)}")
        return

# Test the function
if __name__ == "__main__":
    # Test English
    speak_response("Hello, how are you?", "english")
    
    # Test Malayalam
    speak_response("നമസ്കാരം", "malayalam")