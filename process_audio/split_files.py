#!/usr/bin/python3.6
from pydub import AudioSegment
from pydub.silence import split_on_silence

sound_file = AudioSegment.from_wav("../wav-files/kimche+stoyan-wav/stoyan-3.wav")
audio_chunks = split_on_silence(sound_file,
    # must be silent for at least half a second
    min_silence_len=500,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-50
)

for i, chunk in enumerate(audio_chunks):
    out_file = ".//splitAudio//stoyan//chunk{0}.wav".format(i+151)
    print("exporting", out_file)
    chunk.export(out_file, format="wav")
