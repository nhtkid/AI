pip install elevenlabs
pip install openai
pip install gradio
pip install requests

import gradio as gr
import subprocess
import requests
import openai
import config
openai.api_key = config.OpenAI_API_Key

conversation = [
        {"role": "system", "content": "You are a sensual and caring therapist."}
        ]

def transcribe(audio):
    print ('1st', conversation)

#   Whisper API to transcribe user's query
    audio_file= open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

#   return user query and append to converation variable
    conversation.append({"role": "user", "content": transcript["text"]})
    print ('2nd', conversation)

#   Calling ChatGPT API with user query
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=conversation,
    temperature=0.8,  # Controls the randomness of the responses. Higher values (e.g., 0.8) make the output more random and creative.
    max_tokens=50,  # Limits the length of the response to 50 tokens.
    top_p=0.9,  # Controls the diversity of the responses. Higher values (e.g., 0.9) allow a wider range of tokens to be selected.
)

    system_message = response["choices"][0]["message"]["content"]

#   return the model response and append to the conversation variable
    conversation.append({"role": "assistant", "content": system_message})
    print ('3rd', conversation)

#   Convert text into speech using Eleven Labs API as Audio Stream
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/bqGlZCwvVCZyrtYzMnSx/stream"

    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": config.ELEVEN_API_KEY
    }

    data = {
      "text": system_message,
      "model_id": "eleven_monolingual_v1",
      "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
      }
    }

    response = requests.post(url, json=data, headers=headers, stream=True)
  
#   Use FFPlay to play the stream
    cmd = ['ffplay', '-autoexit', '-']
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for chunk in response.iter_content(chunk_size=1024):
        proc.stdin.write(chunk)

    proc.stdin.close()
    proc.wait()

# Format the conversation for display
    formatted_conversation = ""
    for message in conversation:
        if message["role"] == "user":
            formatted_conversation += "Me: " + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_conversation += "You: " + message["content"] + "\n"
    return formatted_conversation.strip()

#   Launch the app
bot = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text")

bot.launch(debug=True)
