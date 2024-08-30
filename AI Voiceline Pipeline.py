import asyncio
import whisper
import numpy as np
import webrtcvad
import sounddevice as sd
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import edge_tts
import soundfile as sf

# Load the Whisper model
whisper_model = whisper.load_model("small.en")

# Define GPU device or use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load GPT-Neo model and tokenizer
gpt_model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name, clean_up_tokenization_spaces=True)
gpt_model = GPTNeoForCausalLM.from_pretrained(gpt_model_name).to(device)

# Convert model to half precision if using GPU
gpt_model.half()

# Function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Settings for audio capture
sampling_rate = 16000
duration = 5  # Duration to record (in seconds)

# Function to capture audio from the microphone
def record_audio(duration, sampling_rate):
    print("Recording...")
    audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    return audio_data.flatten()

# Convert float32 audio to int16
def convert_float32_to_int16(audio_data):
    return np.int16(audio_data * 32767)

# Convert int16 audio back to float32
def convert_int16_to_float32(audio_data):
    return np.float32(audio_data / 32767)

# Function to capture and process audio with VAD
def capture_and_process_audio():
    audio_data = record_audio(duration, sampling_rate)
    audio_data_int16 = convert_float32_to_int16(audio_data)

    vad = webrtcvad.Vad(2)  # Medium sensitivity
    frame_duration = 20  # ms
    frame_size = int(sampling_rate * frame_duration / 1000)
    frames = [audio_data_int16[i:i + frame_size] for i in range(0, len(audio_data_int16), frame_size)]

    voiced_frames = [frame for frame in frames if len(frame) == frame_size and vad.is_speech(frame.tobytes(), sample_rate=sampling_rate)]

    if voiced_frames:
        voiced_audio_int16 = np.concatenate(voiced_frames)
        voiced_audio = convert_int16_to_float32(voiced_audio_int16)
    else:
        print("No speech detected.")
        voiced_audio = np.array([])  # Empty array if no speech detected

    return voiced_audio

# Function to generate formal and exact definitions using GPT model
def generate_definition(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
    clear_gpu_memory()  # Clear GPU memory before inference
    with torch.no_grad():
        try:
            outputs = gpt_model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=100,  # Adjusted length for detailed definitions
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                do_sample=False,  # Disable sampling for exact and formal definitions
                temperature=0.0,  # Set to 0 for deterministic output
                top_k=50,  # Limit top-k for focused outputs
                top_p=1.0,  # No nucleus sampling
                early_stopping=True
            )
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to convert text to speech using edge-tts
async def text_to_speech(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

# Function to play the generated audio file
def play_audio(file_path):
    data, fs = sf.read(file_path)
    sd.play(data, fs)
    sd.wait()

# Main pipeline function
def voice_query_pipeline():
    # Step 1: Capture and process audio
    voiced_audio = capture_and_process_audio()

    # Step 2: Use Whisper model for speech-to-text if audio is available
    if voiced_audio.size > 0:
        whisper_result = whisper_model.transcribe(voiced_audio, language="en", fp16=False)
        transcribed_text = whisper_result["text"]
        print("Transcribed Text:", transcribed_text)

        # Step 3: Generate response using GPT model
        generated_text = generate_definition(f"Define {transcribed_text} in formal terms.")
        print("Generated Response:", generated_text)

        # Step 4: Convert the generated text to speech
        voice = 'en-US-GuyNeural'  # Select the voice
        output_file = "response.mp3"
        print("Converting text to speech...")
        asyncio.run(text_to_speech(generated_text, voice, output_file))

        # Step 5: Play the generated speech
        print("Playing the response...")
        play_audio(output_file)
    else:
        print("No audio available for transcription.")

if __name__ == "__main__":
    voice_query_pipeline()
