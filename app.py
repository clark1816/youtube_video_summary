import openai
import whisper 
from api_key import API_KEY
from pytube import YouTube

import ffmpeg 
import os
import numpy as np 
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components

openai.api_key = API_KEY

option = st.sidebar.selectbox("Which Dashboard?", ('home', 'YouTube Video Summarizer', 'Contact'),1)
if option == 'home':
    st.title("Welcome to my Dashboard!")
    st.write("This is a dashboard that I built using Streamlit. I hope you enjoy it!")
    st.write("Please select a dashboard from the sidebar.")
    st.write('contact me on linkedin to hire me linked.com/in/abhishek-kumar-1a1b2b1b1')
    
if option == 'YouTube Video Summarizer':
    if "yt_video.mp3" in os.listdir():
        os.remove("yt_video.mp3")
    if "yt_video.mp4" in os.listdir():
        os.remove("yt_video.mp4")
    st.title("YouTube Video Summarizer")
    st.write("This is a dashboard that I built using Streamlit. I hope you enjoy it!")
    url = st.text_input("Enter the URL of the YouTube Video")
    start_time = st.number_input("Enter the start time of the video (in seconds)")
    end_time = st.number_input("Enter the end time of the video (in seconds)")
    if st.button("Summarize"):
        st.write("Summarizing...")
        yt = YouTube(url)
        streams = yt.streams.filter(only_audio=True)
        stream = streams.first()
        stream.download(filename="yt_video.mp4")
        ffmpeg.input('yt_video.mp4').output('yt_video.mp3', ss=start_time, to=end_time).run()
        model = whisper.load_model("base")
        response = model.transcribe('yt_video.mp3')
        st.write(response['text'])
        
        transcript = response['text']
        words = transcript.split(" ")
        chunks = np.array_split(words, 6)
        sentences = ' '.join(list(chunks[0]))
        summary_responses = []
        
        for chunk in chunks:
            
            sentences = ' '.join(list(chunk))

            prompt = f"{sentences}\n\ntl;dr:"

            response = openai.Completion.create(
                engine="text-davinci-003", 
                prompt=prompt,
                temperature=0.3, # The temperature controls the randomness of the response, represented as a range from 0 to 1. A lower value of temperature means the API will respond with the first thing that the model sees; a higher value means the model evaluates possible responses that could fit into the context before spitting out the result.
                max_tokens=150,
                top_p=1, # Top P controls how many random results the model should consider for completion, as suggested by the temperature dial, thus determining the scope of randomness. Top P’s range is from 0 to 1. A lower value limits creativity, while a higher value expands its horizons.
                frequency_penalty=0,
                presence_penalty=1
            )

            response_text = response["choices"][0]["text"]
            summary_responses.append(response_text)

        full_summary = "".join(summary_responses)

        st.header("full summary")
        st.write(full_summary)