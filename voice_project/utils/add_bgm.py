"""
Background music processing module for the Voice Assistant.

This module provides functionality to add background music to text-to-speech
audio files. It handles audio mixing, volume adjustment, and ensures proper
synchronization between the voice and music tracks.

Dependencies:
    - pydub: For audio processing and mixing
"""

from pydub import AudioSegment

def create_tts_with_music(temp_tts_file, output_file="output.mp3", music_file="bgm.mp3", 
                        music_volume=-15):
    """
    Create a text-to-speech audio file with background music.
    
    This function:
    1. Loads the TTS audio and background music
    2. Adjusts the background music volume
    3. Ensures the music track is long enough
    4. Mixes the voice and music tracks
    5. Exports the final audio file
    
    Args:
        temp_tts_file (str): Path to the temporary TTS audio file
        output_file (str, optional): Path for the output file. Defaults to "output.mp3"
        music_file (str, optional): Path to the background music file. Defaults to "bgm.mp3"
        music_volume (int, optional): Volume adjustment for background music in dB.
            Negative values reduce volume. Defaults to -15
    
    Returns:
        str: Path to the output audio file
    
    Raises:
        FileNotFoundError: If the background music file is not found
    
    Note:
        The background music will be looped if it's shorter than the voice track
    """
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
    
    return output_file
