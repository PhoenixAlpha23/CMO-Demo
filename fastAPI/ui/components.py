import base64

def create_audio_player_html(audio_data, auto_play=False):
    """Create custom HTML audio player with auto-play option"""
    if not audio_data:
        return ""
    audio_base64 = base64.b64encode(audio_data).decode()
    autoplay_attr = "autoplay" if auto_play else ""
    
    html = f"""
    <audio controls {autoplay_attr} style="width: 100%; margin: 10px 0;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    return html