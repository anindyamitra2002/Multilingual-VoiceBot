import streamlit as st
import tempfile
import librosa
import soundfile as sf
import os
from lang_detect_client import send_request as detect_language
from asr_client import transcribe_audio
from translator_client import translate_text
from tts_client_api import tts_client
from audiorecorder import audiorecorder
import numpy as np
import io
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Common languages across APIs (expanded for more flexibility)
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

def get_answer(question, huggingface_token):
    """Generate an answer using Hugging Face LLM"""
    template = """
    You are a knowledgeable assistant. Answer the question accurately and concisely within 100 words.

    Question: {question}

    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            temperature=0.2,
            huggingfacehub_api_token=huggingface_token,
            allow_reuse=True
        )
        
        llm_chain = prompt | llm
        return llm_chain.invoke({"question": question})
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def save_audio_file(uploaded_audio):
    """Save the uploaded audio file as a temporary WAV file"""
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    y, sr = librosa.load(uploaded_audio, sr=None)
    sf.write(temp_wav.name, y, sr)
    return temp_wav.name

def handle_voice_pipeline(audio_file, input_lang, output_lang, out_voice, huggingface_token):
    """Handle the full voice input pipeline"""
    # Step 1: Detect language if input language is "Detect Automatically"
    # if input_lang == "Detect Automatically":
    #     audio_array, sr = librosa.load(audio_file)
    #     detected_lang = detect_language(audio_array)
    #     st.write(f"Detected Language: {detected_lang}")
    #     input_lang = lang_dict.get(detected_lang, None)
    
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
    answer_in_english = get_answer(translated_text, huggingface_token)
    st.write(f"Answer: {answer_in_english}")
    
    # Step 5: Translate answer back to the output language
    if COMMON_LANGUAGES[output_lang] != "eng_Latn":
        translated_answer = translate_text(answer_in_english, "eng_Latn", COMMON_LANGUAGES[output_lang])
    else:
        translated_answer = answer_in_english
    st.write(f"Translated Answer: {translated_answer}")
    
    # Step 6: Convert the final answer to speech using TTS
    audio_output_file = tts_client(translated_answer, output_lang.lower(), out_voice)
    st.audio(audio_output_file, format="audio/wav", autoplay=True)
    return translated_answer

def handle_text_pipeline(question, input_language, output_language, output_voice_gender, huggingface_token):
    """Handle text-based chat pipeline with language detection and translation"""
    # Step 1: Detect input language if not specified
    if input_language == "Detect Automatically":
        input_language = detect_input_language(question)
    
    # Step 2: Translate question to English if not already in English
    if COMMON_LANGUAGES[input_language] != "eng_Latn":
        translated_question = translate_text(
            question, 
            COMMON_LANGUAGES[input_language], 
            "eng_Latn"
        )
    else:
        translated_question = question
    
    # Step 3: Get answer in English from LLM
    answer_in_english = get_answer(translated_question, huggingface_token)
    
    # Step 4: Translate answer to output language
    if COMMON_LANGUAGES[output_language] != "eng_Latn":
        translated_answer = translate_text(
            answer_in_english, 
            "eng_Latn", 
            COMMON_LANGUAGES[output_language]
        )
    else:
        translated_answer = answer_in_english
    
    # Step 5: Convert to speech if needed
    audio_output_file = tts_client(
        translated_answer, 
        output_language.lower(), 
        output_voice_gender
    )
    
    return translated_answer, audio_output_file

def main():
    # Page configuration
    st.set_page_config(
        page_title="🌐 Multilingual Assistant", 
        page_icon="🤖", 
        layout="wide"
    )

    # Sidebar for configuration
    with st.sidebar:
        st.title("🛠️ Assistant Setup")
        
        # Hugging Face Token Input
        huggingface_token = st.text_input("🔑 Hugging Face Token", type="password")
        
        # Mode Selection with Emojis
        mode = st.radio("📝 Choose Interaction Mode", 
            ["🎙️ Voice Input", "💬 Text Chat"],
            index=0
        )
        
        # Language Selection with Flags
        st.subheader("🌍 Language Settings")
        input_language = st.selectbox("🔠 Input Language", 
            list(COMMON_LANGUAGES.keys()),
            index=list(COMMON_LANGUAGES.keys()).index("Bengali")
        )
        output_language = st.selectbox("🗣️ Output Language", 
            list(COMMON_LANGUAGES.keys()),
            index=list(COMMON_LANGUAGES.keys()).index("Hindi")
        )
        

        output_voice_gender = st.radio("🎤 Output Voice", 
            ["Female", "Male"], 
            index=0
        )


    # Main Content Area
    st.title("🌐 Multilingual Voice & Chat Assistant")

    # Initialize session state for messages only when needed
    if mode == "💬 Text Chat":
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display chat history only in text chat mode
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    # Voice Input Mode
    if mode == "🎙️ Voice Input":
        st.subheader("🎤 Record Your Question")
        audio = audiorecorder("🔴 Click to Record", "⏹️ Stop Recording")

        if len(audio) > 0:
            # Play recorded audio
            st.audio(audio.export().read())  
            audio.export("input_audio.wav", format="wav")

        # Process button
        if st.button("🚀 Process Voice Input") and huggingface_token:
            with st.spinner("🔍 Processing voice input..."):
                try:
                    # Run voice pipeline
                    answer = handle_voice_pipeline(
                        "input_audio.wav", 
                        input_language, 
                        output_language,
                        output_voice_gender.lower(), 
                        huggingface_token
                    )
                except Exception as e:
                    st.error(f"❌ Error processing voice input: {str(e)}")

    # Text Chat Mode
    if mode == "💬 Text Chat":
        st.subheader("💬 Ask me Anything")
        
        # Chat input
        if prompt := st.chat_input("✍️ Type your message here"):
            if not huggingface_token:
                st.warning("🔒 Please enter your Hugging Face token in the sidebar.")
                st.stop()
            
            # Display user message
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner(text="🤖 Generating response..."):
                try:
                    # Process text input
                    answer, audio_file = handle_text_pipeline(
                        prompt, 
                        input_language,
                        output_language, 
                        output_voice_gender.lower(),  # default voice gender for text chat
                        huggingface_token
                    )
                    
                    # Display assistant response
                    st.chat_message("assistant").write(answer)
                    
                    st.audio(audio_file, format="audio/wav", autoplay=False)
                    
                    # Update chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"❌ Error processing text input: {str(e)}")

if __name__ == "__main__":
    main()