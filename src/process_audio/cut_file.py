#!/usr/bin/python3.6
from pydub import AudioSegment
import wave
import sys

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def trimmed_audio(sound):
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)
    trimmed_sound = sound[start_trim:duration-end_trim]
    return trimmed_sound

def main(file_name):
    w = AudioSegment.from_wav(file_name)

    current_time = 0;
    t = 5 * 1000
    counter = 0
    while True:
        new_audio = w[current_time:current_time+t]
        raw_duration = new_audio.duration_seconds

        new_audio = trimmed_audio(new_audio)
        new_audio_name = file_name[:-4] + "." + str(counter) + ".wav"
        new_audio.export(new_audio_name, format="wav")

        current_time += t
        counter += 1
        print(raw_duration)
        if(raw_duration < t/1000):
            break

if __name__ == "__main__":
   main(sys.argv[1])
