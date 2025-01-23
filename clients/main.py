import streamlit as st
import tempfile
import librosa
import soundfile as sf
import os
from lang_detect_client import send_request as detect_language
from asr_client import transcribe_audio
from translator_client import translate_text
# from llm_client import get_answer
from tts_client_api import tts_client
from audiorecorder import audiorecorder
import numpy as np
import librosa
from pydub import AudioSegment
import io

import os
from langchain_huggingface import HuggingFaceEndpoint
# from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACEHUB_API_TOKEN = "your_secret_token_here"

# Function to generate an answer for a given question
def get_answer(question):
    template = """
    You are a knowledgeable assistant. Answer the question accurately and concisely within 100 words.

    Question: {question}

    Answer:
    """
    
    # Create the prompt template
    prompt = PromptTemplate.from_template(template)

    # Model repository ID and initialization of the Hugging Face endpoint
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.2,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        allow_reuse=True
    )
    
    # Chain the prompt with the LLM
    llm_chain = prompt | llm
    
    # Invoke the LLM chain with the question and return the answer
    return llm_chain.invoke({"question": question})

# # Example usage
# question = "Who won the FIFA World Cup in the year 1994?"
# answer = get_answer(question)
# print(answer)

# Common languages across APIs
COMMON_LANGUAGES = {
    'Bengali': 'ben_Beng',
    'English': 'eng_Latn',
    'Gujarati': 'guj_Gujr',
    'Hindi': 'hin_Deva',
    'Kannada': 'kan_Knda',
    'Malayalam': 'mal_Mlym',
    'Marathi': 'mar_Deva',
    'Odia': 'ory_Orya',
    'Punjabi': 'pan_Guru',
    'Tamil': 'tam_Taml',
    'Telugu': 'tel_Telu',
    'Urdu': 'urd_Arab'
}

lang_dict = {
    "asm": "Assamese",
    "ben": "Bengali",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "odi": "Odia",
    "pun": "Punjabi",
    "tam": "Tamil",
    "tel": "Telugu",
    "eng": "English"
}

def save_audio_file(uploaded_audio):
    """
    Save the uploaded audio file as a temporary WAV file using librosa and return the file path.
    """
    # Create a temporary file to store the audio
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    
    # Load the audio using librosa
    y, sr = librosa.load(uploaded_audio, sr=None)
    
    # Write the audio to the temporary file using soundfile
    sf.write(temp_wav.name, y, sr)
    
    return temp_wav.name

def handle_pipeline(audio_file, input_lang):
    """Handle the full pipeline: detect language (if necessary), transcribe, translate, process via LLM, translate back, and generate speech."""
    print(audio_file)
    # Step 1: Detect language if input language is "Detect Automatically"
    if input_lang == "Detect Automatically":
        audio_array, sr = librosa.load(audio_file)
        detected_lang = detect_language(audio_array)
        st.write(f"Detected Language: {detected_lang}")
        input_lang = lang_dict.get(detected_lang, None)
    
    # Step 2: Transcribe audio using ASR API
    transcription = transcribe_audio(audio_file, input_lang)
    st.write(f"Transcribed Text: {transcription}")
    
    # Step 3: Translate transcription to English (if not already in English)
    if COMMON_LANGUAGES[input_lang] != "eng_Latn":
        translated_text = translate_text(transcription, COMMON_LANGUAGES[input_lang], "eng_Latn")
    else:
        translated_text = transcription
    st.write(f"Translated Text to English: {translated_text}")
    
    # Step 4: Get an answer from the LLM
    answer_in_english = get_answer(translated_text)
    st.write(f"Answer: {answer_in_english}")
    
    # Step 5: Translate answer back to the input language
    if COMMON_LANGUAGES[input_lang] != "eng_Latn":
        translated_answer = translate_text(answer_in_english, "eng_Latn", COMMON_LANGUAGES[input_lang])
    else:
        translated_answer = answer_in_english
    st.write(f"Translated Answer: {translated_answer}")
    print(f"Translated Answer: {translated_answer}")
    # Step 6: Convert the final answer to speech using TTS
    audio_output_file = tts_client(translated_answer, input_lang.lower(), "female")
    print("audio_output_file: ", audio_output_file)
    st.audio(audio_output_file, format="audio/wav", autoplay=True)

# Streamlit app UI
st.title("Multilingual Voice Assistant")
st.write("Upload a voice file and choose the input language to get answers to your question.")

audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    # To play audio in frontend:
    st.audio(audio.export().read())  

    # To save audio to a file, use pydub export method:
    audio.export("input_audio.wav", format="wav")

# Input language selection
input_language = st.selectbox("Select input language or choose 'Detect Automatically'", 
                              ["Detect Automatically"] + list(COMMON_LANGUAGES.keys()))
btn = st.button("Process")
# Handle file processing and pipeline execution
if len(audio) > 0 and btn:
    with st.spinner("Processing..."):
        temp_audio_path = "input_audio.wav"
        
        # Pass the temporary wav file path to the pipeline
        handle_pipeline(temp_audio_path, input_language)
        
        # Clean up the temporary file after processing
        # os.remove(temp_audio_path)
