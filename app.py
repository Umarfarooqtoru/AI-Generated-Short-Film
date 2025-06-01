import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import torch
from gtts import gTTS
import os
import uuid
import ffmpeg
from io import BytesIO

# --- Page Config ---
st.set_page_config(page_title="🎬 AI Short Film Generator", layout="wide")

# --- Title ---
st.title("🎬 AI Short Film Generator")
st.write("Generate a short film using AI (no APIs required!)")

# --- Sidebar (Settings) ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Text Generation Model",
        ["GPT-2", "LLaMA (Coming Soon)"],  # Add more options later
    )
    duration_per_scene = st.slider("Seconds per scene", 2, 10, 5)

# --- Main App ---
prompt = st.text_input(
    "Enter your film idea:",
    placeholder="e.g., A robot learning to dance in a cyberpunk city",
)

if st.button("Generate Film 🎥"):
    if not prompt:
        st.error("Please enter a prompt!")
    else:
        with st.spinner("Generating your film..."):
            # --- Step 1: Generate Script ---
            st.subheader("📜 Script")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(**inputs, max_length=300)
            script = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(script)

            # --- Step 2: Generate Images ---
            st.subheader("🎨 Generating Scenes...")
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            scenes = script.split(".")[:3]  # Extract 3 key scenes
            image_paths = []
            for i, scene in enumerate(scenes):
                if not scene.strip():
                    continue
                image = pipe(scene).images[0]
                img_path = f"scene_{i}.png"
                image.save(img_path)
                image_paths.append(img_path)
                st.image(image, caption=f"Scene {i+1}: {scene}")

            # --- Step 3: Generate Voiceover ---
            st.subheader("🔊 Voiceover")
            tts = gTTS(script[:500], lang='en')  # Limit to 500 chars for speed
            audio_path = "voice.mp3"
            tts.save(audio_path)
            st.audio(audio_path)

            # --- Step 4: Assemble Video ---
            st.subheader("🎥 Final Film")
            output_video = "film_output.mp4"
            
            # FFmpeg: Combine images + audio
            ffmpeg.input(
                "concat:" + "|".join(image_paths),
                framerate=1/duration_per_scene,
            ).output(
                output_video,
                vcodec="libx264",
                pix_fmt="yuv420p",
                acodec="aac",
            ).overwrite_output().run()
            
            # Display video
            st.video(output_video)

            # Download button
            with open(output_video, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="Download Film 📁",
                data=video_bytes,
                file_name="ai_film.mp4",
                mime="video/mp4",
            )

st.markdown("---")
st.caption("Built by UMER Farooq)")
