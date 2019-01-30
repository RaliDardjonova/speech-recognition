#!/usr/bin/python3.6
import sys
import wave
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from pydub import AudioSegment
from pydub.silence import split_on_silence


def get_windows(stream, window_size):
    little_window = int(window_size*(3.0/4.0))
    X = []
    x = read(stream, window_size)
    if len(x) != window_size:
        return
    X.extend(x)
    yield np.array(X)
    X = X[little_window:]
    while True:
        x = read(stream, little_window)
        if len(x) != little_window:
            break
        X.extend(x)
        #print len(X)
        yield np.array(X)
        X = X[little_window:]

def read(stream, window_size):
    num_channels = stream.getnchannels()
    sample_width = stream.getsampwidth()

    x = stream.readframes(window_size)
    x = np.fromstring(x, dtype=np.int8 if sample_width == 1 else np.int16)

    x = np.reshape(x, (-1, num_channels))

    if num_channels > 1:
        x = (x[:,0] + x[:,1]) / 2
    else:
        x = x[:,0]
    return x

def get_class(index, word, all_count):
    word_len = len(word)
    class_index = (float(index)/float(all_count))*word_len
    #print all_count
    #print index
    #print class_index
    return word[int(class_index)]


#w = wave.open('its-not-that-easy.wav', 'r')
def get_features(filename, word):
    w = wave.open(filename, 'r')
    window_size = 1024
    sample_width = w.getsampwidth()
    sample_rate  = w.getframerate()

    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(window_size)) / window_size)

    Y = []
    for x in get_windows(w, window_size):
        y = np.fft.rfft(x*hann)
        #print len(y)
        #print np.fft.rfftfreq(1024)
        Y.append(y)

    Y = np.column_stack(Y)
    Y = Y.transpose()
    # tuk za wseki red da wzimame razlichnite featur-i
    # w edin cikyl  - za wseki red se wzimat featu-rite i se slagat w now masiv
    # kydeto na wseki red shte imame nqkolko feature-a i klasa(bukwata)
    # sreden ygyl, maks ygyl, sredna amplituda, maks amplituda i na koe mqsto, intenzitet, stand. otklonenie
    #print Y
    #print 1
    window_count = Y.shape[0]
    print(window_count)
    features = []
    index = 0;
    for index, chunk in enumerate(Y):
        print(index)

        feature = []
        chunk_len = len(chunk)
        angles = np.angle(chunk)
        amplitudes = np.absolute(chunk)
        #amplitudes = amplitudes/np.max(amplitudes) ako shte normirame do 1?

        intensities = amplitudes*amplitudes

        ampl_argmax = float(np.argmax(amplitudes))/chunk_len
        max_amplitude = np.max(amplitudes)
        angle_argmax = float(np.argmax(angles))/chunk_len
        angle_max = np.max(angles)

        ampl_mean = np.mean(amplitudes)
        int_mean = np.mean(intensities)
        angle_mean = np.mean(angles)
        ampl_std = np.sqrt(np.mean(abs(amplitudes - ampl_mean)**2))
        int_std = np.sqrt(np.mean(abs(intensities - int_mean)**2))
        angle_std = np.sqrt(np.mean(abs(angles - angle_mean)**2))

        feature.append(get_class(index, word, window_count))
        feature.append(ampl_argmax)
        feature.append(max_amplitude)
        feature.append(angle_argmax)
        feature.append(angle_max)
        feature.append(ampl_mean)
        feature.append(int_mean)
        feature.append(angle_mean)
        feature.append(ampl_std)
        feature.append(int_std)
        feature.append(angle_std)

        features.append(feature)
    w.close()
    return features

word_file = open("words.txt", 'r')
words = word_file.read().splitlines()

shuffled_words = [256, 255, 254, 252, 251, 250,249,248, 247, 245, 244, 242, 241,
                240, 239, 238, 237, 236, 235, 234, 233, 232, 231,230, 229, 228,
                227, 226, 225, 223, 222, 221, 220, 219, 218, 217, 216, 215, 213,
                212, 211, 210, 209, 208, 207, 205, 204, 203, 202, 201, 200, 199,
                198, 196, 195, 194, 192, 191, 190, 189, 188, 187, 186, 185, 184,
                183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172 , 170,
                168, 167, 166, 165, 164, 163, 162, 161, 160, 157, 154, 153, 152,
                151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139,
                138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126,
                125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 113, 112,
                111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99,
                98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83,
                82, 81, 80, 79, 77, 76, 75, 74, 73, 72, 71, 70, 68, 67, 66, 65,
                64, 63, 62, 61, 60, 59, 56, 54, 53, 52, 51, 50, 49, 48, 47,
                46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31,
                30, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
                13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
'''
sound_file = AudioSegment.from_wav(sys.argv[1])
audio_chunks = split_on_silence(sound_file,
    # must be silent for at least half a second
    min_silence_len=500,

    # consider it silent if quieter than -16 dBFS : -35 momiche, -40 kimche, -45 stoyan
    silence_thresh=-35
)
print "*"
'''
for i, filename in enumerate(os.listdir(".//splitAudio")):
    rel_filename = ".//splitAudio/" + filename
    features = get_features(rel_filename, words[shuffled_words[i]-1])
    write_file = open(".//features//features-gerasim-momiche2.txt", "a")
    write_file.write("\n"+words[shuffled_words[i]-1]+"\n")
    for row in features:
        for f in row:
            write_file.write(str(f)+",")
        write_file.write("\n")
