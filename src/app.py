import validators
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import openai
import requests
import pandas as pd

# Streamlit app
st.subheader('₿ SoundCoin:')
st.caption("Select you favorite crypto and favorite song and listen to the market ")

# Get OpenAI API key and URL to be summarized
with st.sidebar:

    
    openai_api_key= st.text_input("OpenAI API key", value="", type="password")
    st.caption(
        "*If you don't have an OpenAI API key, get it [here](https://platform.openai.com/account/api-keys).*")
    model = st.selectbox("OpenAI chat model",
                         ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"))
    st.caption("*If the article is long, choose gpt-3.5-turbo-16k.*")

crypto=st.select_slider("Select a crypto asset", options=['BTC', 'ETH', 'XRP', 'ADA'])
#url = st.text_input("Youtube URL for your favorite song", label_visibility="collapsed")

# If 'Create song' button is clicked
if st.button("Create song"):
    # Validate inputs
    if not openai_api_key.strip():# or not url.strip():
        st.error("Please provide the missing fields.")
    #elif not validators.url(url):
    #    st.error("Please enter a valid URL.")
    else:
        openai.api_key=openai_api_key
        try:
            with st.spinner("Please wait..."):

                # fetch crypto data from binance
                endpoint = 'https://api.binance.com/api/v1/klines'
                symbol = crypto+'USDT'
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

                st.line_chart(data=df,x='timestamp', y='close')

                # Load URL data
                #loader = UnstructuredURLLoader(urls=[url])
                #data = loader.load()

                # Step 2: convert price action into text

                
                prompt_template="""Generate a text description of the recent price action for {crypto} against USDT over the past 4 hours 
                based on the following last 4 OHLCV {crypto}/USDT 1h candles: {data}.
                """

                prompt_template = PromptTemplate.from_template(prompt_template)
                prompt=prompt_template.format(crypto=crypto, data=df[-4:].to_string())
                text = openai.Completion.create(engine="text-davinci-003", prompt=prompt,temperature=0,max_tokens=500).choices[0].text.strip()

                st.success(text)

                prompt_template2 = """Write a 1 sentence song description, specifying instruments and style, from the following text:

                    {text}

                """

                prompt_template2 = PromptTemplate.from_template(prompt_template2)
                prompt2=prompt_template2.format(text=text)
                song_description = openai.Completion.create(engine="text-davinci-003", prompt=prompt2,temperature=0,max_tokens=500).choices[0].text.strip()

                st.info(song_description)
                # Load the MusicGen model
                processor = AutoProcessor.from_pretrained(
                    "facebook/musicgen-small")
                model = MusicgenForConditionalGeneration.from_pretrained(
                    "facebook/musicgen-small")

                # Format the input based on the song description
                inputs = processor(
                    text=[song_description],
                    padding=True,
                    return_tensors="pt",
                )

                # Generate the audio
                audio_values = model.generate(**inputs, max_new_tokens=1536)

                sampling_rate = model.config.audio_encoder.sampling_rate
                # Save the wav file into your system
                scipy.io.wavfile.write(
                    "musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

                # Render a success message with the song description generated by ChatGPT
                st.success("Your song has been succesfully created with the following prompt: "+song_description)
        except Exception as e:
            st.exception(f"Exception: {e}")
