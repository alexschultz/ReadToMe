import boto3
import sys
from contextlib import closing
from playsound import playsound

def speak(text):
    client = boto3.client('polly', region_name='us-east-1')
    response = client.synthesize_speech(
        OutputFormat='mp3',
        Text=text,
        TextType='text',
        VoiceId='Brian'
    )
    if "AudioStream" in response:
        with closing(response["AudioStream"]) as stream:
            try:
                file = open('/tmp/speech.mp3', 'wb')
                file.write(stream.read())
                file.close()
                playAudioFile('/tmp/speech.mp3')
            except IOError as error:
                print(error)
                sys.exit(-1)


def playAudioFile(path):
    playsound(path, True)