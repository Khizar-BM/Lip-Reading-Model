import streamlit as st
import os
import tensorflow as tf
from utils import load_data, num_to_char, make_gif
from modelutil import load_model
import imageio

st.set_page_config(layout="wide")


with st.sidebar:
    st.image("https://res-3.cloudinary.com/fieldfisher/image/upload/c_lfill,dpr_1,g_auto,h_700,w_1240/f_auto,q_auto/v1/sectors/technology/tech_neoncircuitboard_857021704_medium_lc5h05")
    st.title("LipNet")
    st.info("This application follows the LipNet deep learning paper.")

st.title("LipNet Streamlit App")

options = os.listdir(os.path.join("data", "s1"))
selected_video = st.selectbox("Select video", options)

col1, col2 = st.columns(2)

if selected_video:
    file_path = os.path.join("data", "s1", selected_video)
    roi, marked_video, annotation = load_data(tf.convert_to_tensor(file_path))

    with col1:
        st.info("Following is the raw video converted to mp4 format")

        os.system(f"ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y")

        # video = open("app/test_video.mp4", "rb")
        # video_bytes = video.read()
        # st.video(video_bytes)
        st.info("Below is the actual annotation of the video")
        st.text(tf.strings.reduce_join(num_to_char(annotation.numpy())).numpy().decode('utf-8'))

    with col2:
        st.info("The mouth region is extracted from the video using dlib")
        make_gif(roi, "roi.gif")
        imageio.v3.imwrite("marked.gif", marked_video, loop=True)

        st.image("marked.gif")

        st.info("Our deep learning model only sees the grayscale extracted mouth region")
        st.image("roi.gif", width=320)

        model = load_model()
        yhat = model.predict(tf.expand_dims(roi, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True )[0][0].numpy()
        st.info("This is the output of the model as tokens")
        st.text(decoder)

        st.info("Decode the raw tokens into words")
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)









