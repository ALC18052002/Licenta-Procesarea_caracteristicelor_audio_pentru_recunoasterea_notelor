import json
import os
import math
import librosa
import numpy as np

DATASET_PATH = "path/to/marsyas/dataset"
JSON_WRITE_PATH = "/Users/alexanlazar/Licenta-Procesarea_caracteristicelor_audio_pentru_recunoasterea_notelor/dataset/json_test_write.json"
AUDIO_PATH = '/Users/alexanlazar/Licenta-Procesarea_caracteristicelor_audio_pentru_recunoasterea_notelor/dataset/maestro-v3.0.0_2/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav'
JSON_READ_PATH = '/Users/alexanlazar/Licenta-Procesarea_caracteristicelor_audio_pentru_recunoasterea_notelor/dataset/json_test.json'
SAMPLE_RATE = 22050
HOP_LENGTH = 512
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
N_FFT = 2048

def calculate_frame_indexes(start_sample, finish_sample):
    start_frame = int(start_sample / HOP_LENGTH)
    end_frame = int(finish_sample / HOP_LENGTH)
    return start_frame, end_frame

spectral_centroid = []
bandwidth = []

def fancy_amplitude_envelope(signal, frame_size, hop_length):
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store mapping, labels, and MFCCs
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
    with open(JSON_READ_PATH) as json_file:
        label_data = json.load(json_file)

    
    signal, sample_rate = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE)

    duration = int(int(int(librosa.get_duration(y = signal, sr=sample_rate)) * SAMPLE_RATE /HOP_LENGTH)*HOP_LENGTH)
  
    # calculate start and finish sample for current segment
    start = 0
    finish = duration - 5 * HOP_LENGTH

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
    # store only mfcc feature with expected number of vectors

    data["input"]["mfcc"].append(mfcc.tolist())
    data["input"]["centroid"].append(spectral_centroid.tolist())
    data["input"]["bandwidth"].append(bandwidth.tolist())
    data["input"]["amplitude"].append(amplitude.tolist())
    data["input"]["zcr"].append(zcr.tolist())
    data["input"]["rms"].append(rms.tolist())
    data["input"]["fft"].append(fft[0].tolist())
    data["input"]["chroma"].append(chroma.tolist())
    data["labels"].append(label_data[start_frame:end_frame])

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
if __name__ == "__main__":

    save_mfcc(DATASET_PATH, JSON_WRITE_PATH )
