import requests
import json
import base64

def tts_client(text, lang, gender, alpha=1, output_file="tts_output.wav"):
    """
    A TTS client function that sends text to the server and retrieves speech audio.

    :param text: The text to be converted to speech
    :param lang: The language of the text (e.g., 'hindi', 'bengali', etc.)
    :param gender: The voice gender ('male' or 'female')
    :param alpha: (Optional) Speed control (default is 1 for normal speed)
    :param output_file: The output WAV file path to save the speech (default is 'tts_output.wav')
    
    :return: The path to the saved audio file
    """

    # API endpoint
    url = "http://localhost:5000/tts"  # Update this if your TTS server URL is different

    # Create the payload for the request
    payload = json.dumps({
        "input": text,
        "gender": gender,
        "lang": lang,
        "alpha": alpha  # Speed control, default is normal speed
    })

    headers = {'Content-Type': 'application/json'}
    
    try:
        # Send the POST request to the TTS server
        response = requests.post(url, headers=headers, data=payload).json()
        print("response: ", response)
        if 'audio' not in response:
            raise ValueError("No audio data returned from the server")

        # Save the received encoded audio to a WAV file
        audio = response['audio']
        with open(output_file, 'wb') as wav_file:
            wav_file.write(base64.b64decode(audio))

        print(f"Audio file saved as {output_file}")
        return output_file

    except Exception as e:
        print(f"Error during TTS request: {e}")
        return None

# # Example usage
# if __name__ == "__main__":
#     text = "বিকেল ৫টা বাজলেও সুপ্রিম কোর্টের নির্দেশ মোতাবেক কর্মবিরতি প্রত্যাহার করলেন না জুনিয়র চিকিৎসকেরা। স্বাস্থ্য ভবনের সামনে অবস্থানে বসে রয়েছেন তাঁরা। দাবি না মানা হলে সেখানেই বসে থাকার ডাক দিয়েছেন তাঁরা। ইতিমধ্যেই স্বাস্থ্য ভবনের সামনে এসে পৌঁছেছে ভ্রাম্যমান শৌচাগার। স্বাস্থ্য ভবনের একটি সূত্র মারফত জানা গিয়েছিল, চিকিৎসকদের একটি প্রতিনিধি দল ভিতরে এসে কথা বলতে চাইলে তাঁদের স্বাগত জানানো হবে। এই বিষয়ে জুনিয়র ডাক্তারদের বক্তব্য, তাঁরা ডেপুুটেশন দিতে আসেননি। তাঁদের দাবি স্পষ্ট। সেগুলি মানা না হলে লাগাতার অবস্থান চলবে।"  # Hindi text example
#     lang = "bengali"
#     gender = "female"
#     audio_file = tts_client(text, lang, gender, alpha=1, output_file="output_tts.wav")

#     if audio_file:
#         print(f"Speech generated and saved in: {audio_file}")
