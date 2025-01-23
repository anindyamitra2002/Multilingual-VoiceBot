import requests
import numpy as np
import wave

def load_audio_file(audio_path):
    """Loads a .wav file and returns the audio data as a numpy array."""
    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        audio_data = wav_file.readframes(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    return audio_array

def send_request(audio_array):
    """Send a request with an audio array to the litserve API."""
    url = "http://127.0.0.1:8001/predict"  # Adjust the URL to your server's endpoint

    # Prepare the request payload
    payload = {
        "audio_array": audio_array.tolist()  # Convert array to a list for JSON
    }

    # Send the POST request to the server
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Language: {result['predicted_language']}")
        print(f"Probabilities: {result['probabilities']}")
        return result['predicted_language']
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None

# if __name__ == "__main__":
#     # Test audio file path
#     audio_file_path = "samples/q1.wav"  # Replace with the actual file path
    
#     # Load the audio file into a numpy array
#     audio_array = load_audio_file(audio_file_path)
    
#     # Send the audio array to the API for language prediction
#     send_request(audio_array)
