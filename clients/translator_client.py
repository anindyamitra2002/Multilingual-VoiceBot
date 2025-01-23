import requests

def translate_text(text, input_lang, target_lang):
    """
    Function to translate text from one language to another using the translation API.
    
    :param text: The text to translate
    :param input_lang: The language of the input text (e.g., 'eng_Latn' for English)
    :param target_lang: The language to translate the text into (e.g., 'ben_Beng' for Bengali)
    
    :return: Translated text if successful, error message otherwise.
    """
    # Define the API endpoint (assuming it's running on localhost:8000)
    url = "http://127.0.0.1:8000/translate"
    
    # Define the payload with input text, input language, and target language
    payload = {
        "text": text,
        "input_lang": input_lang,
        "target_lang": target_lang
    }
    
    try:
        # Send the POST request
        response = requests.post(url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Return the translated text
            return response.json()["translated_text"]
        else:
            # Return error message if the request failed
            return f"Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        # Handle network-related errors
        return f"An error occurred: {e}"

# Example usage
# if __name__ == "__main__":
#     text = "When I was young, I used to go to the park every day. We watched a new movie last week, which was very inspiring."
#     input_language = "eng_Latn"
#     target_language = "ben_Beng"

#     translated_text = translate_text(text, input_language, target_language)
#     print("Translated Text:", translated_text)
