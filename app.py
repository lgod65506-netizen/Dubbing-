import gradio as gr
import whisper
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from transformers import pipeline
from pydub import AudioSegment
import os
import subprocess

# Global variables
whisper_model = whisper.load_model("base")
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", device=0)
tts_model = ChatterboxTTS.from_pretrained(device="cuda")

def dub_video(video_path, emotion_level):
    if not video_path:
        return None, "Pehle video upload karein!"
    
    # 1. Audio extract karein
    audio_path = "temp_audio.wav"
    subprocess.run(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y", shell=True)
    
    # 2. Reference Voice (10 sec)
    audio = AudioSegment.from_file(audio_path)
    ref_path = "ref_voice.wav"
    audio[:10000].export(ref_path, format="wav")
    
    # 3. Whisper se text nikalein
    result = whisper_model.transcribe(audio_path, language="ja")
    
    # 4. NLLB se Hindi translate karein
    final_audio = AudioSegment.empty()
    
    for seg in result["segments"]:
        text = seg["text"].strip()
        if len(text) < 2: continue
        
        hindi_text = translator(text, src_lang="jpn_Jpan", tgt_lang="hin_Deva")[0]['translation_text']
        
        # 5. Chatterbox se aawaz banayein
        wav = tts_model.generate(
            hindi_text,
            audio_prompt_path=ref_path,
            exaggeration=emotion_level,
            cfg_weight=0.3
        )
        line_path = f"line_{seg['start']}.wav"
        ta.save(line_path, wav, tts_model.sr)
        
        # Audio ko jodein
        line_audio = AudioSegment.from_wav(line_path)
        final_audio += line_audio
        os.remove(line_path)

    # 6. Final audio save karein
    final_audio.export("dubbed_audio.wav", format="wav")
    
    # 7. Video mein audio add karein
    subprocess.run(f"ffmpeg -i {video_path} -i dubbed_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest final_dubbed.mp4 -y", shell=True)
    
    return "final_dubbed.mp4", "Dubbing poori ho gayi!"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Anime Hindi Dubbing Tool")
    with gr.Row():
        video_input = gr.Video(label="Japanese Anime Clip Upload Karein")
    with gr.Row():
        emotion_slider = gr.Slider(0.1, 1.0, value=0.7, label="Emotion Level")
    with gr.Row():
        btn = gr.Button("Dub Karein")
    with gr.Row():
        video_output = gr.Video(label="Hindi Dubbed Video")
        status = gr.Textbox(label="Status")

    btn.click(dub_video, inputs=[video_input, emotion_slider], outputs=[video_output, status])

demo.launch()
