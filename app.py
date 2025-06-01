import streamlit as st
from transformers import pipeline

# Configure the page
st.set_page_config(page_title="AI Script Generator", layout="centered")
st.title("📜 AI Script Generator")

# Simple text input
prompt = st.text_input(
    "Enter your film idea:",
    placeholder="e.g., A robot learning to dance",
    key="prompt"
)

if st.button("Generate Script"):
    if not prompt:
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Generating your script..."):
            # Use a small text-generation model
            generator = pipeline("text-generation", model="distilgpt2")
            generated_text = generator(
                prompt,
                max_length=300,
                num_return_sequences=1
            )[0]["generated_text"]
            
            st.subheader("Generated Script")
            st.write(generated_text)
            
            # Optional: Add download button
            st.download_button(
                label="Download Script",
                data=generated_text,
                file_name="ai_script.txt",
                mime="text/plain"
            )

st.caption("Built with DistilGPT2 - No APIs required")
