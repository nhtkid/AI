pip install openai
pip install gradio
pip install requests

import gradio as gr
import openai
import config
openai.api_key = config.OpenAI_API_Key

conversation = [
        {"role": "system", "content": "You are a human companion and advisor for life and work"}
            ]

def transcribe(audio):
    
#   Whisper API
    audio_file= open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    
#   return transcript["text"]

    conversation.append({"role": "user", "content": transcript["text"]})


#   ChatGPT API

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=conversation
    )

    system_message = response["choices"][0]["message"]["content"]

#   return system_message

    conversation.append({"role": "assistant", "content": system_message})


#   Synthesizes speech from the input string of text
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file('path/to/google/api/credential')
    from google.cloud import texttospeech

#   Language detection
    import langdetect
    detected_lang = langdetect.detect(transcript["text"])
    
#   Define a dictionary to map the detected language to language code and voice name
    language_dict = {
        "fr": ("fr-FR", "fr-FR-Neural2-A"),
        "ko": ("ko-KR", "ko-KR-Wavenet-A"),
        "ja": ("ja-JP", "ja-JP-Neural2-D"),
    }

#   Set the language and voice for Google TTS based on the detected language
    if detected_lang in language_dict:
        language_code, voice_name = language_dict[detected_lang]
    else:
        language_code = "en-GB"
        voice_name = "en-GB-Neural2-D"

    client = texttospeech.TextToSpeechClient(credentials=credentials)
    
    input_text = texttospeech.SynthesisInput(text=system_message)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )


    # The response's audio_content is binary.
    import uuid
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')


    return "output.mp3"

bot = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="audio")

bot.launch(debug=True)
