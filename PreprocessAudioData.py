import json
import os
import math
import librosa
import numpy as np

DATASET_PATH = "/Users/alexanlazar/Licenta-Procesarea_caracteristicelor_audio_pentru_recunoasterea_notelor/dataset/maestro-v3.0.0_2"
JSON_WRITE_PATH = "/Users/alexanlazar/Licenta-Procesarea_caracteristicelor_audio_pentru_recunoasterea_notelor/dataset/json_test_write.json"
JSON_READ_PATH = '/Users/alexanlazar/Licenta-Procesarea_caracteristicelor_audio_pentru_recunoasterea_notelor/dataset/midi_to_json.json'
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


def save_mfcc(dataset_path, json_path, num_mfcc=39, n_fft=2048, hop_length=512):
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
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            print("\nProcessing: {}".format(semantic_label))

            if(semantic_label == '2015'):
            # process all audio files in genre sub-dir
                for f in filenames:
            # load audio file
                    file_path = os.path.join(dirpath, f)
                    extension = file_path.split(".")
                    extension = extension[len(extension) - 1]

                    if extension == 'wav':

                        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

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
                        path = file_path[0: len(file_path) - 4]
                        path+= ".midi"
                        data["labels"].append(label_data[path]['labels'][start_frame:end_frame + 1])

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_WRITE_PATH)
