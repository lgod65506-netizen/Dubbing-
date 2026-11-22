import streamlit as st
import whisper
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydub import AudioSegment
import os
import subprocess
import tempfile
import torch

st.set_page_config(page_title="Anime Hindi Dubber", layout="centered")
st.title("🎬 Anime Hindi Dubber")
st.write("Apni Japanese anime clip upload karein aur Hindi mein dub karein!")

@st.cache_resource
def load_models():
    # 1. Whisper Model
    whisper_model = whisper.load_model("base")
    
    # 2. Translation Model (Naya tarika jisse error nahi aayega)
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 3. Chatterbox TTS Model
    # Streamlit free tier par GPU nahi hota, isliye "cpu" use kar rahe hain
    tts_model = ChatterboxTTS.from_pretrained(device="cpu") 
    
    return whisper_model, tokenizer, translator_model, tts_model

st.info("Models load ho rahe hain, thoda time lagega...")
whisper_model, tokenizer, translator_model, tts_model = load_models()
st.success("Models taiyaar hain!")

uploaded_file = st.file_uploader("Video Upload Karein (MP4)", type=["mp4", "mkv", "mov"])

if uploaded_file is not None:
    if st.button("Dubbing Shuru Karein"):
        with st.spinner("Video process ho rahi hai... (Thoda time lagega)"):
            
            # 1. Temp file save karein
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_file.read())
                video_path = temp_video.name
                
            # 2. Audio extract karein
            audio_path = "temp_audio.wav"
            subprocess.run(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y", shell=True)
            
            # 3. Reference Voice nikalein (Voice cloning ke liye pehle 10 second)
            audio = AudioSegment.from_file(audio_path)
            ref_path = "ref_voice.wav"
            audio[:10000].export(ref_path, format="wav")
            
            # 4. Whisper se Japanese text nikalein
            st.text("Japanese audio se text nikala ja raha hai...")
            result = whisper_model.transcribe(audio_path, language="ja")
            
            # 5. Translate aur Generate
            st.text("Hindi translation aur aawaz ban rahi hai...")
            final_audio = AudioSegment.empty()
            
            for seg in result["segments"]:
                text = seg["text"].strip()
                if len(text) < 2: continue
                
                # Naya Translation Code
                tokenizer.src_lang = "jpn_Jpan"
                inputs = tokenizer(text, return_tensors="pt")
                translated_tokens = translator_model.generate(
                    **inputs, 
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"), 
                    max_length=512
                )
                hindi_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                # Aawaz banayein
                wav = tts_model.generate(
                    hindi_text,
                    audio_prompt_path=ref_path,
                    exaggeration=0.7,
                    cfg_weight=0.3
                )
                line_path = f"line_{seg['start']}.wav"
                ta.save(line_path, wav, tts_model.sr)
                
                line_audio = AudioSegment.from_wav(line_path)
                
                # Original timing ke hisaab se sync karne ki koshish
                silence_duration = int((seg['start'] - len(final_audio)/1000)*1000)
                if silence_duration > 0:
                    silence = AudioSegment.silent(duration=silence_duration)
                    final_audio += silence
                    
                final_audio += line_audio
                os.remove(line_path)

            final_audio.export("dubbed_audio.wav", format="wav")
            
            # 6. Video mein nayi aawaz add karein
            final_video_path = "final_dubbed.mp4"
            subprocess.run(f"ffmpeg -i {video_path} -i dubbed_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest {final_video_path} -y", shell=True)
            
            st.success("Dubbing poori ho gayi!")
            
            # 7. Download button
            with open(final_video_path, "rb") as file:
                st.download_button(
                    label="Dubbed Video Download Karein",
                    data=file,
                    file_name="hindi_dubbed.mp4",
                    mime="video/mp4"
                )
