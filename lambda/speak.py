import boto3
import io
import os
import sys
from contextlib import closing
from pydub import AudioSegment
from pydub.playback import play

def speak(text):
    client = boto3.client('polly', region_name='us-east-1')
    response = client.synthesize_speech(
        OutputFormat='mp3',
        Text=text,
        TextType='text',
        VoiceId='Brian'
    )

    print(response)

    if "AudioStream" in response:
        with closing(response["AudioStream"]) as stream:
            try:
                data = stream.read()
                song = AudioSegment.from_file(io.BytesIO(data), format="mp3")
                play(song)
            except IOError as error:
                print(error)
                sys.exit(-1)


def playAudioFile(path):
    data = open(os.path.realpath(path), 'rb').read()
    song = AudioSegment.from_file(io.BytesIO(data), format="mp3")
    play(song)