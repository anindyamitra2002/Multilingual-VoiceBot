import litserve as ls
import os
import numpy as np
from scipy.io.wavfile import write
from LIDv2.detect_lang import initialize_model, predict_language_wav
# Assuming the previous model setup functions are imported
# from the previously written code

class LanguageIdentifierAPI(ls.LitAPI):
    def setup(self, device):
        # Initialize the language identification model and the evaluator
        self.lang_model, self.evaluator = initialize_model()

    def decode_request(self, request):
        # The audio array will be passed as part of the request
        audio_array = np.array(request["audio_array"], dtype=np.float32)
        
        # Write the audio array to a temporary wav file
        temp_wav_path = "temp_audio.wav"
        write(temp_wav_path, 16000, audio_array)  # Assuming sample rate of 16000 Hz
        return temp_wav_path

    def predict(self, audio_path):
        # Use the predict_language_wav function to predict the language
        predicted_language, probabilities = predict_language_wav(self.lang_model, self.evaluator, audio_path)
        
        # Clean up the temporary file after processing
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return predicted_language, probabilities

    def encode_response(self, output):
        predicted_language, probabilities = output
        
        # Returning the predicted language and probabilities as JSON response
        return {
            "predicted_language": predicted_language,
            "probabilities": probabilities.tolist()  # Convert to list for JSON compatibility
        }

if __name__ == "__main__":
    api = LanguageIdentifierAPI()
    server = ls.LitServer(api, accelerator="auto", workers_per_device=2)
    server.run(port=8001)
