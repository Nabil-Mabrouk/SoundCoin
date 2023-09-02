import streamlit as st
import requests
import pandas as pd
from moviepy.editor import *
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
import scipy
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import openai
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip, CompositeVideoClip, TextClip, AudioFileClip
from moviepy.video.io.bindings import mplfig_to_npimage

def fetch_crypto_data(crypto):
    """Fetch market data for the selected cryptocurrency and calculate indicators."""
    endpoint = 'https://api.binance.com/api/v1/klines'
    symbol = crypto + 'USDT'
    interval = '1h'  # 1-hour candles
    limit = 1000  # Number of data points to retrieve

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }

    # Make the API request
    response = requests.get(endpoint, params=params)
    data = response.json()

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Select and convert columns to floats
    df = df[['timestamp', 'open', 'close', 'high', 'low', 'volume']].astype({'open': float, 'close': float, 'high': float, 'low': float, 'volume': float})

    # Calculate indicators
    df['short_term_volatility'] = df['close'].rolling(window=2).std()
    df['long_term_volatility'] = df['close'].rolling(window=10).std()
    
    # Calculate short-term trend (simple moving average over 10 periods)
    df['short_term_trend'] = df['close'].rolling(window=2).mean()
    
    # Calculate long-term trend (simple moving average over 50 periods)
    df['long_term_trend'] = df['close'].rolling(window=10).mean()
    
    return df


def generate_song_description(crypto, style, df):
    """Generate a text description of the recent price action."""
    prompt_template = """I will provide you with a dataframe the 1h price action of crypto {crypto} formed by
    100 OCHLV candles and long and short terms volatilities and trends. 
    Your role is to Generate a text description of the recent price action. Focus on short term compared 
    to long term trends and short term variability compared to long term variability: the short_term_volatility 
    the long_term_volatility, the short_term_trend and the long_term_trend. Your text should describe a music 
    that mimick the market behaviour. Don't explain your answer. Only provide a description of a 30 seconds song by specifying 
    the dynamic the style will be {style} style. You answer should be on the following format: ' A song ### song description'.
    
    The dataframe: {data}
    
    """
    prompt_template = PromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(crypto=crypto, style=style, data=df[-20:].to_string())
    text = openai.Completion.create(engine="text-davinci-003", prompt=prompt, temperature=0, max_tokens=2000).choices[0].text.strip()
    
    return text

#def generate_song_description(text):
#    """Generate a 1-sentence song description."""
#    prompt_template = """Write a 1 sentence song description, specifying instruments and style, from the following text:
#
#    {text}
#    """
#    prompt_template = PromptTemplate.from_template(prompt_template)
#    prompt = prompt_template.format(text=text)
#    song_description = openai.Completion.create(engine="text-davinci-003", prompt=prompt, temperature=0, max_tokens=500).choices[0].text.strip()
#    
#    return song_description

def create_audio(song_description):
    """Create audio based on the song description."""
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    inputs = processor(
        text=[song_description],
        padding=True,
        return_tensors="pt",
    )
    audio_values = model.generate(**inputs, max_new_tokens=1536)
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())


def create_video(df, symbol):
    """Create a video showing dynamic price action with audio and timestamp."""
    
    # Define the duration and frame rate
    duration = 30
    fps = 20
    
    def make_frame(t):
        # Calculate the index based on time to create a dynamic drawing
        index = int(len(df) * t / duration)
        data_to_plot = df.iloc[:index + 1]

        # Create a Matplotlib figure
        fig, ax = plt.subplots()
        ax.plot(data_to_plot['timestamp'], data_to_plot['close'], lw=3)
        
        # Add a timestamp to the figure
        timestamp_text = f'Time: {data_to_plot.iloc[-1]["timestamp"]}'
        ax.text(0.05, 0.9, timestamp_text, transform=ax.transAxes, fontsize=12, color='white')

        return mplfig_to_npimage(fig)

    # Create the video clip with dynamic drawing
    clip = VideoClip(make_frame, duration=duration)

    # Load the audio clip and add a fade-out effect
    audio = AudioFileClip("musicgen_out.wav")
    #audio = audio.fadeout(2)  # Add a 2-second fade-out at the end

    # Combine audio and video clips
    clip = clip.set_audio(audio)

    cat1 = (ImageClip("SoundCoin.png")
            .set_start(0) #which second to start displaying image
            .set_duration(4) #how long to display image
            .set_position(("center", "center")))

    final_clip=CompositeVideoClip([clip, cat1])

    file_name = symbol+".mp4"

    final_clip.write_videofile(file_name, fps=20) 

    return file_name

def main():
    st.title('₿ SoundCoin:')
    st.info("Select you favorite crypto and your favorite music style and SoundCoin will use AI to generate a short Videoclip showing the recent price action with AudioCraft AI generated Song")
    st.image('soundcoin.jpg', use_column_width=True) 

    # Get OpenAI API key and URL to be summarized
    with st.sidebar:

        openai_api_key= st.text_input("OpenAI API key", value="", type="password")
        st.caption(
            "*If you don't have an OpenAI API key, get it [here](https://platform.openai.com/account/api-keys).*")
        model = st.selectbox("OpenAI chat model",
                         ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"))




    crypto = st.selectbox("Select a crypto",("BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL","TRX","DOT","MATIC", "LTC", "SHIB"))
    style= st.selectbox("Select a music style",("RAP", "HIP-HOP", "DRILL", "DANCE", "ROCK", "BLUES", "CLASSIC","HARD ROCK"))

    if st.button("Create"):
        # Validate inputs
        if not openai_api_key.strip():# or not url.strip():
            st.error("Please provide the missing fields.")
        else:
            openai.api_key=openai_api_key

        # Fetch crypto data
        df = fetch_crypto_data(crypto)

        # Generate text description of price action
        st.subheader("Step 1: Fetch market data")
        st.line_chart(data=df[-20:], x='timestamp', y='close')

        song_description = generate_song_description(crypto, style, df)

        st.subheader("Step 2: Convert the price action into instructions to AudioCraft")
        st.info(song_description)

        create_audio(song_description)

        st.subheader("Step 3: Create the video and mix it with the AI generated audio")
        st.info("Creating the #short video clip ..")

        filename=create_video(df, crypto + 'USDT')

        video_file=open(filename, 'rb')
        video_bytes = video_file.read()

        st.subheader("View your creation :-)")

        st.video(video_bytes)

        st.subheader("Step 5: Export your video to youtube #shorts")
        st.button("Export")
        st.success("visit : https://www.youtube.com/@NabilMabrouk-pc5wp/")

if __name__ == "__main__":
    main()
