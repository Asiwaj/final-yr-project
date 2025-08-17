# First, ensure you have the necessary libraries installed using uv.
# uv pip install transformers sounddevice scipy torch cohere sentencepiece yarngpt torchaudio

import sounddevice as sd
import numpy as np
import torch
import cohere
import time
import os # Import the os module to read environment variables and check paths
from os.path import isdir # Import isdir to check if a path is a directory
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoModelForSeq2SeqLM
import torchaudio
from yarngpt import generate_speech # Use the simple generate_speech function

# --- Configuration ---
# Set the path to your downloaded models.
# The paths have been updated to use absolute paths to prevent errors.
# The paths are still relative to the location of the main.py file.

ASR_MODEL_PATH = os.path.abspath("../models/afrospeech-wav2vec-all-6")
TRANSLATION_MODEL_PATH = os.path.abspath("../models/m2m100_418M")

# Get the Cohere API key from an environment variable.
# You MUST set this variable before running the script.
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Check if the API key was found.
if not COHERE_API_KEY:
    raise ValueError("❌ Cohere API key not found. Please set the COHERE_API_KEY environment variable.")

# Audio recording parameters
FS = 16000  # Sample rate for recording (16kHz is standard for ASR models)
DURATION = 5  # Duration of recording in seconds
CHANNELS = 1 # Mono audio
DTYPE = 'float32' # Data type for audio array
TTS_SAMPLE_RATE = 24000 # Sample rate for YarnGPT output

# --- Model Loading ---
def load_models():
    """Loads the ASR, Translation models, and sets up the Cohere client."""
    print("Loading models...")
    
    # Print the absolute paths for debugging
    print(f"Checking for ASR model at: {ASR_MODEL_PATH}")
    print(f"Checking for Translation model at: {TRANSLATION_MODEL_PATH}")

    # Check if the ASR model path exists before loading
    if not os.path.exists(ASR_MODEL_PATH) or not isdir(ASR_MODEL_PATH):
        print(f"❌ ASR model path not found or is not a directory: {ASR_MODEL_PATH}")
        print("Please check your file path and update the ASR_MODEL_PATH variable.")
        return None, None, None, None, None
    
    # Check if the Translation model path exists before loading
    if not os.path.exists(TRANSLATION_MODEL_PATH) or not isdir(TRANSLATION_MODEL_PATH):
        print(f"❌ Translation model path not found or is not a directory: {TRANSLATION_MODEL_PATH}")
        print("Please check your file path and update the TRANSLATION_MODEL_PATH variable.")
        return None, None, None, None, None

    # Load ASR processor
    try:
        asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_PATH, local_files_only=True)
        # asr_processor = AutoProcessor.from_pretrained("chrisjay/afrospeech-wav2vec-all-6", local_files_only=True)
        print("✅ ASR processor loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading ASR processor from local path: {e}")
        print("This is often caused by missing or improperly downloaded files.")
        return None, None, None, None, None
        
    # Load ASR model
    try:
        asr_model = AutoModelForCTC.from_pretrained(ASR_MODEL_PATH, local_files_only=True)
        print("✅ ASR model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading ASR model from local path: {e}")
        print("This is often caused by missing or improperly downloaded files.")
        return None, None, None, None, None
    
    # Load Translation model and tokenizer
    try:
        translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_PATH)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_PATH)
        print("✅ Translation model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading Translation model from local path: {e}")
        return None, None, None, None, None

    # Set up Cohere client
    try:
        co = cohere.Client(COHERE_API_KEY)
        # Verify the API key by making a simple request
        _ = co.chat(model="command-a-03-2025", message="Hello", stream=False)
        print("✅ Cohere client initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing Cohere client: {e}. Check your API key and network connection.")
        return None, None, None, None, None

    # YarnGPT model loading is handled by the library.
    print("✅ YarnGPT library is ready to use.")

    return asr_processor, asr_model, translation_tokenizer, translation_model, co

# --- Audio Functions ---
def record_audio(duration=DURATION):
    """
    Records audio from the microphone for a specified duration.
    Returns the audio data as a NumPy array.
    """
    print(f"👂 Listening for {duration} seconds...")
    audio_data = sd.rec(int(duration * FS), samplerate=FS, channels=CHANNELS, dtype=DTYPE)
    sd.wait() # Wait until recording is finished
    print("Recording finished.")
    return audio_data

def transcribe_igbo(audio_data, asr_processor, asr_model):
    """
    Transcribes the recorded audio data into Igbo text using the ASR model.
    """
    print("🧠 Transcribing audio...")
    input_values = asr_processor(audio_data.squeeze(), sampling_rate=FS, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = asr_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(predicted_ids)[0]
    
    print(f"✅ Transcription (Igbo): {transcription}")
    return transcription

# --- Translation and LLM Functions ---
def translate_igbo_to_english(igbo_text, tokenizer, model):
    """Translates Igbo text to English."""
    print("🔄 Translating Igbo to English...")
    tokenizer.src_lang = "ig"
    encoded_igbo = tokenizer(igbo_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_igbo, forced_bos_token_id=tokenizer.get_lang_id("en"))
    english_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"✅ Translation (English): {english_text}")
    return english_text

def generate_cohere_response(english_text, co):
    """Generates a text response in English using the Cohere LLM."""
    print("🤔 Generating response with Cohere...")
    try:
        response = co.chat(
            model="command-a-03-2025",
            message=english_text
        )
        llm_response = response.text
        print(f"✅ Cohere Response (English): {llm_response}")
        return llm_response
    except Exception as e:
        print(f"❌ Error generating Cohere response: {e}")
        return "Sorry, I am unable to generate a response at this time."

def translate_english_to_igbo(english_text, tokenizer, model):
    """Translates English text to Igbo."""
    print("🔄 Translating English to Igbo...")
    tokenizer.src_lang = "en"
    encoded_english = tokenizer(english_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_english, forced_bos_token_id=tokenizer.get_lang_id("ig"))
    igbo_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"✅ Translation (Igbo): {igbo_text}")
    return igbo_text

def speak_igbo_response(igbo_text):
    """
    Converts the Igbo text response to speech and plays it using YarnGPT.
    """
    print("🗣️ Converting text to speech and playing with YarnGPT...")
    try:
        # Use the generate_speech function directly from the yarngpt library
        # The library handles the model loading and generation internally.
        audio_data = generate_speech(igbo_text, language="igbo", speaker="chioma")
        
        # Convert the audio tensor to a NumPy array for playback
        audio_data_np = audio_data.numpy()
        
        # Play the audio
        sd.play(audio_data_np, samplerate=TTS_SAMPLE_RATE)
        sd.wait()
        
        print(f"🔊 Playing Igbo audio: {igbo_text}")

    except Exception as e:
        print(f"❌ Error in text-to-speech: {e}")

# --- Main Program Loop ---
def main():
    asr_processor, asr_model, translation_tokenizer, translation_model, co = load_models()
    
    if not all([asr_processor, asr_model, translation_tokenizer, translation_model, co]):
        print("\nCannot start the assistant due to model loading or API errors.")
        return

    print("\nIgbo AI Assistant is ready! Say something in Igbo. Press Ctrl+C to exit.")
    
    while True:
        try:
            # 1. Listen for audio
            audio_data = record_audio()
            
            # 2. Transcribe the audio from Igbo speech to Igbo text
            igbo_text_in = transcribe_igbo(audio_data, asr_processor, asr_model)
            
            # 3. Translate the Igbo text to English
            english_text_for_llm = translate_igbo_to_english(igbo_text_in, translation_tokenizer, translation_model)
            
            # 4. Generate a response in English with Cohere
            english_response = generate_cohere_response(english_text_for_llm, co)
            
            # 5. Translate the English response back to Igbo
            igbo_text_out = translate_english_to_igbo(english_response, translation_tokenizer, translation_model)
            
            # 6. Speak the Igbo response using YarnGPT
            speak_igbo_response(igbo_text_out)
            
            print("\n-----------------------------------\n")
            
        except KeyboardInterrupt:
            print("\nExiting the assistant. Dalụ!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
