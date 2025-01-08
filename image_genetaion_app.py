import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import time

# Set Streamlit layout at the start
st.set_page_config(page_title="Text-to-Image Generator", layout="wide")

# Initialize the model
@st.cache_resource
def load_model():
    # Load Stable Diffusion model
    device = "cpu"  # Force CPU usage
    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
    )
    model.to(device)
    return model

# Resize image to fit within a maximum width
def resize_image_to_fit(image, max_width=600):
    width, height = image.size
    if width > max_width:
        ratio = max_width / float(width)
        new_height = int(ratio * height)
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return image

# Main app function
def main():
    st.title("✨ Text-to-Image Generator with Stable Diffusion")
    st.markdown("Generate stunning images from text using **Stable Diffusion 2.1**.")

    # Sidebar for input parameters
    st.sidebar.header("Customize Your Image")
    prompt = st.sidebar.text_area(
        "Enter your prompt:", 
        placeholder="e.g., A serene sunset over a mountain lake", 
        height=150
    )
    num_inference_steps = st.sidebar.slider(
        "Number of Inference Steps:",
        min_value=10,
        max_value=50,
        value=40,
        step=5,
        help="Higher values improve image quality but take longer."
    )
    guidance_scale = st.sidebar.slider(
        "Guidance Scale:",
        min_value=5.0,
        max_value=20.0,
        value=8.5,
        step=0.5,
        help="Controls how closely the image follows the prompt."
    )
    
    # Button to generate the image
    if st.sidebar.button("Generate Image"):
        if prompt.strip():
            with st.spinner("Generating your image..."):
                # Load the model (cached for performance)
                model = load_model()

                # Start timing for performance measurement
                start_time = time.time()

                # Generate the image using the Stable Diffusion model
                output = model(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=512,
                    width=512,
                )
                image = output.images[0]

                # Resize the image to fit within the screen
                resized_image = resize_image_to_fit(image)

                # Calculate generation time
                time_taken = round(time.time() - start_time, 2)

                # Display the generated image
                st.subheader("Generated Image:")
                st.image(resized_image, caption=f"Generated in {time_taken}s", use_container_width=True)

        else:
            st.error("Please enter a prompt to generate an image.")

    # Footer
    st.markdown(
        "<div style='text-align: center; margin-top: 20px;'>Made with ❤️ using Stable Diffusion</div>",
        unsafe_allow_html=True,
    )

# Run the app
if __name__ == "__main__":
    main()
