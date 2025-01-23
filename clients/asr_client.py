import requests
import os
from pydub import AudioSegment

def convert_to_wav(audio_path):
    """Converts the audio file to wav format if not already a wav."""
    file_extension = os.path.splitext(audio_path)[1].lower()
    if file_extension not in ['.mp3', '.mp4', '.wav', '.ogg']:
        audio = AudioSegment.from_file(audio_path)
        wav_path = os.path.splitext(audio_path)[0] + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path
    return audio_path

def transcribe_audio(audio_path, input_lang):
    """
    Function to transcribe audio using the provided ASR API.
    
    :param audio_path: Path to the audio file (will be converted to wav if not already).
    :param input_lang: Language of the input audio.
    :return: Transcribed text.
    """
    # Convert the audio to wav format if needed
    # audio_path = convert_to_wav(audio_path)
    input_lang = input_lang.lower()
    # Prepare the request payload
    print("input_lang: ", input_lang)
    print("audio_file: ", audio_path)
    files = {
        'file': open(audio_path, 'rb'),
        'language': (None, input_lang),
        'vtt': (None, 'true'),  # Use 'vtt': (None, 'false') if you don't want VTT format
    }

    # Send the POST request to the ASR API
    response = requests.post('https://asr.iitm.ac.in/internal/asr/decode', files=files)
    res = response.json()
    print(res)
    # Check for a successful response
    if response.status_code == 200:
        return response.json().get("transcript", "No transcription available")
    else:
        raise Exception(f"ASR request failed with status code {response.status_code}: {response.text}")

# Example usage
if __name__ == "__main__":
    audio_file = "samples/q2.wav"  # Replace with your audio file path
    input_language = "bengali"  # Replace with the language of the audio

    try:
        transcription = transcribe_audio(audio_file, input_language)
        print("Transcription:", transcription)
    except Exception as e:
        print("Error:", str(e))
