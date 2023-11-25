"""Streamlit app to detect components in images."""

import logging

import streamlit as st
import pandas as pd

import aws
import gpt

# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)


# Define functions
def detect_objects(image_file, components):
    """Detect objects in images."""
    with spinner_placeholder:
        hashed_image, hashed_image_name = aws.hash_and_scale_image(
            mode="file", image_file=image_file
        )
        if not aws.find_image(hashed_image_name):
            image_url = aws.upload_image(
                mode="file", image_name=hashed_image_name, image_file=hashed_image
            )
        else:
            image_url = f"https://{aws.BUCKET}.s3.amazonaws.com/{hashed_image_name}"
        labels = gpt.detect_labels(image_url, components)
        logging.info(labels)
        st.session_state.labels = labels


# Configure Streamlit page and state
st.set_page_config(page_title="Image Tagger", page_icon="🤖")
if "labels" not in st.session_state:
    st.session_state.labels = []
if "error" not in st.session_state:
    st.session_state.error = ""

# Render Streamlit page
st.title("Tag image")
st.markdown("""Upload an image and define the components you want to detect.""")

image_file = st.file_uploader(
    label="Upload image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

components = st.text_input(
    label="Components (comma-separated list)",
    value="Main text, Other text, Main object, Other objects, Background, Color palette, Style",
)

st.button(
    label="Identify components",
    type="primary",
    on_click=detect_objects,
    args=(image_file, components),
)

spinner_placeholder = st.empty()

if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.labels:
    st.markdown("""---""")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image_file)
    with col2:
        df = pd.DataFrame(
            [label.split(":") for label in st.session_state.labels],
            columns=["Component", "Value"],
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
