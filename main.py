import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import scipy
from sys import argv
import os
from os.path import exists
import tensorflow as tf
import json

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
NEURONAL_NETWORK_PATH = 'neuronal_data_folder/full_model_tf'
AUDIO_FILE_PATH = 'dataset/maestro-v3.0.0_2/2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--2.wav'
UNIQUE_LABELS_CONFIG = 'dataset/Unique_Labels_Config/unique_labels_config.json'
PREDICTION_JSON_PATH = 'dataset/predictions.json'
model = tf.keras.models.load_model(NEURONAL_NETWORK_PATH)

def calculate_frame_indexes(start_sample, finish_sample):
    start_frame = int(start_sample / HOP_LENGTH)
    end_frame = int(finish_sample / HOP_LENGTH)
    return start_frame, end_frame

spectral_centroid = []
bandwidth = []

def fancy_amplitude_envelope(signal, frame_size, hop_length):
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])

def load_data(data_path, num_mfcc=39, n_fft=2048, hop_length=512):

    data = {
        "input":
        {
            "mfcc": [],
            "bandwidth":[],
            "centroid": [],
            "amplitude": [],
            "zcr":[],
            "rms":[],
            "fft":[],
            "chroma":[]
        },
        "labels": []
    }

    signal, sample_rate = librosa.load(data_path, sr=SAMPLE_RATE)

    duration = librosa.get_duration(y=signal, sr = sample_rate)

    start = 0
    
    finish = int(duration * SAMPLE_RATE / N_FFT) * N_FFT

    # extract mfcc
    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal[start:finish], sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    mfcc = mfcc.T
    bandwidth = librosa.feature.spectral_bandwidth(y=signal[start:finish], sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=signal[start:finish], frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y=signal[start:finish], frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    amplitude = fancy_amplitude_envelope(signal[start:finish], N_FFT, HOP_LENGTH)
    start_frame, end_frame = calculate_frame_indexes(start, finish)

    fft = np.abs(librosa.stft(y=signal[start:finish],n_fft=N_FFT, hop_length=HOP_LENGTH))
    chroma = librosa.feature.chroma_stft(S=fft, sr=sample_rate)

    data["input"]["mfcc"].append(mfcc.tolist())
    data["input"]["centroid"].append(spectral_centroid.tolist())
    data["input"]["bandwidth"].append(bandwidth.tolist())
    data["input"]["amplitude"].append(amplitude.tolist())
    data["input"]["zcr"].append(zcr.tolist())
    data["input"]["rms"].append(rms.tolist())
    data["input"]["fft"].append(fft[0].tolist())
    data["input"]["chroma"].append(chroma.tolist())

    mfcc = []
    centroid = np.array(0)
    bandwidth = np.array(0)
    amplitude = np.array(0)
    zcr = np.array(0)
    rms = np.array(0)
    fft = np.array(0)
    chroma = []
    label_sizes = 0

    n = 1

    for i in range ( 0, n):
        amplitude = np.append(amplitude, np.array(data["input"]["amplitude"][i]))

    label_sizes = len(data["input"]["amplitude"][0])

    for j in range(0, len(np.array(data["input"]["chroma"][0]))):
        chroma.append( np.array(data["input"]["chroma"][0][j]))

    for i in range (1, n):
        for j in range(0, len(np.array(data["input"]["chroma"][i]))):
            chroma[j] = np.append(chroma[j] , np.array(data["input"]["chroma"][i][j]))

    for i in range (0, n):
        for j in range(0, len(np.array(data["input"]["chroma"][i]))):
            chroma[j] = chroma[j][:label_sizes]

    for i in range ( 0, n):
        amplitude = amplitude[:label_sizes]
        for j in range(0, len(np.array(data["input"]["mfcc"][i]))):
            mfcc.append( np.array(data["input"]["mfcc"][i][j]))
        mfcc = mfcc[:label_sizes]
        centroid = np.append(centroid, np.array(data["input"]["centroid"][i]))
        centroid = centroid[:label_sizes]
        bandwidth = np.append(bandwidth, np.array(data["input"]["bandwidth"][i]))
        bandwidth = bandwidth[:label_sizes]
        
        zcr = np.append(zcr, np.array(data["input"]["zcr"][i]))
        zcr = zcr[:label_sizes]
        rms = np.append(rms, np.array(data["input"]["rms"][i]))
        rms = rms[:label_sizes]
        fft = np.append(fft, (data["input"]["fft"][i]))
        fft = fft[:label_sizes]
        

    mfcc = np.array(mfcc)
    chroma = np.array(chroma)

    print(mfcc.shape, centroid.shape, bandwidth.shape, amplitude.shape, zcr.shape, rms.shape, fft.shape, chroma.shape)

    x = np.empty((label_sizes, 5 + mfcc.shape[1] + chroma.shape[0]))
    for i in range(0, label_sizes ):
        x[i][0] = centroid[i]
        x[i][1] = bandwidth[i]
        x[i][2] = amplitude[i]
        x[i][3] = zcr[i]
        x[i][4] = rms[i]
        x[i][5] = fft[i]
        for j in range(0, mfcc[0].size):
            x[i][j+5] = mfcc[i][j]
        
        for j in range(0, chroma.shape[0]):
            x[i][j+5+mfcc[0].size] = chroma[j][i]
    return x 
        
if __name__ == "__main__":
    model = tf.keras.models.load_model(NEURONAL_NETWORK_PATH)

    x_pred = load_data(AUDIO_FILE_PATH)

    # Make predictions
    predictions = model.predict(x_pred)

    labels = np.argmax(predictions, axis=1).tolist()


    with open(UNIQUE_LABELS_CONFIG, "r") as fp:
        unique_labels_config = json.load(fp)
    
    label_nr = 0
    for element in labels:
            labels[label_nr] = unique_labels_config["label"][element]
            label_nr = label_nr + 1
    label_nr = 0

    processed_labels = {
        "labels": [],
        "ticks_per_beat": 480,
    }
    processed_labels["labels"] = labels
    with open( PREDICTION_JSON_PATH, "w") as fp:
        json.dump(processed_labels, fp, indent=4)
