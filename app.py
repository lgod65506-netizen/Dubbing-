import streamlit as st
import whisper
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from pydub import AudioSegment
from deep_translator import GoogleTranslator
import subprocess
import tempfile
import os

st.set_page_config(page_title="Anime Hindi Dubber", layout="centered")
st.title("🎬 Anime Hindi Dubber")
st.write("Apni Japanese anime clip upload karein aur Hindi mein dub karein!")

# Models ko sirf tab load karein jab zaroorat ho (RAM bachane ke liye)
@st.cache_resource
def load_models():
    st.info("Models load ho rahe hain, thoda time lagega...")
    whisper_model = whisper.load_model("tiny")
    tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    return whisper_model, tts_model

uploaded_file = st.file_uploader("Video Upload Karein (MP4)", type=["mp4", "mkv", "mov"])

if uploaded_file is not None:
    if st.button("Dubbing Shuru Karein"):
        with st.spinner("Video process ho rahi hai..."):
            
            # 1. Models load karein
            whisper_model, tts_model = load_models()
            
            # 2. Video save karein
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_file.read())
                video_path = temp_video.name
                
            # 3. Audio extract karein
            audio_path = "temp_audio.wav"
            subprocess.run(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y", shell=True)
            
            # 4. Voice Reference (Pehle 10 second)
            audio = AudioSegment.from_file(audio_path)
            ref_path = "ref_voice.wav"
            audio[:10000].export(ref_path, format="wav")
            
            # 5. Whisper se text nikalein
            st.text("Japanese audio se text nikala ja raha hai...")
            result = whisper_model.transcribe(audio_path, language="ja")
            
            # 6. Translate aur Aawaz banayein
            st.text("Hindi translation aur aawaz ban rahi hai...")
            final_audio = AudioSegment.empty()
            translator = GoogleTranslator(source='ja', target='hi')
            
            for seg in result["segments"]:
                text = seg["text"].strip()
                if len(text) < 2: continue
                
                try:
                    hindi_text = translator.translate(text)
                except:
                    hindi_text = text
                
                # Chatterbox se aawaz (Emotion ke saath)
                wav = tts_model.generate(
                    hindi_text,
                    language_id="hi",
                    audio_prompt_path=ref_path,
                    exaggeration=0.7,
                    cfg_weight=0.3
                )
                line_path = f"line_{seg['start']}.wav"
                ta.save(line_path, wav, tts_model.sr)
                
                line_audio = AudioSegment.from_wav(line_path)
                
                # Original timing ke hisaab se sync
                silence_duration = int((seg['start'] - len(final_audio)/1000)*1000)
                if silence_duration > 0:
                    final_audio += AudioSegment.silent(duration=silence_duration)
                    
                final_audio += line_audio
                os.remove(line_path)

            final_audio.export("dubbed_audio.wav", format="wav")
            
            # 7. Video mein aawaz milayein
            final_video_path = "final_dubbed.mp4"
            subprocess.run(f"ffmpeg -i {video_path} -i dubbed_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest {final_video_path} -y", shell=True)
            
            st.success("Dubbing poori ho gayi!")
            
            # 8. Download button
            with open(final_video_path, "rb") as file:
                st.download_button(
                    label="Dubbed Video Download Karein",
                    data=file,
                    file_name="hindi_dubbed.mp4",
                    mime="video/mp4"
                )
