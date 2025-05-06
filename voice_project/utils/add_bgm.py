from pydub import AudioSegment

def create_tts_with_music(temp_tts_file, output_file="output.mp3", music_file="bgm.mp3", 
                        music_volume=-15):
    """
    Create a TTS audio file with background music
    
    Parameters:
    - text: Text to convert to speech
    - output_file: Output file path
    - music_file: Background music file path
    - language: Language for TTS
    - music_volume: Volume adjustment for background music (negative values reduce volume)
    """
    # Create TTS audio
    # temp_tts_file = "temp_response.mp3"
    
    # Load the TTS and background music
    voice = AudioSegment.from_file(temp_tts_file)
    try:
        music = AudioSegment.from_file(music_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Background music file '{music_file}' not found")
    
    
    # Adjust the volume of the background music
    music = music - abs(music_volume)  # Reduce volume by X dB

    
    # Make sure the music is at least as long as the voice
    if len(music) < len(voice):
        # Loop the music if needed
        repeats = int(len(voice) / len(music)) + 1
        music = music * repeats
    
    # Trim music to the length of voice
    music = music[:len(voice)]
    
    # Overlay the voice on top of the music
    combined = music.overlay(voice)
    
    # Export the combined audio
    combined.export(output_file, format="mp3")
    
    # Clean up the temporary file
    
    return output_file
