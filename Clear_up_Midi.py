import mido
import numpy as np
import math
import os
import json

SAMPLE_RATE = 22050
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE
N_FFT = 2048
HOP_LENGTH = 512
DATASET_PATH = 'dataset'



def process_midi_file(midi_file):

    my_mid = mido.MidiFile()


    meta_track = mido.MidiTrack()

    notes = []
    mid = mido.MidiFile(midi_file)
    length = mid.length
    ticks_per_beat = mid.ticks_per_beat
    current_time = 0
    tempo = mido.bpm2tempo(120)
    tempo_msg = mido.MetaMessage('set_tempo', tempo=tempo)
    meta_track.append(tempo_msg)
    my_mid.ticks_per_beat = mid.ticks_per_beat
    my_mid.tracks.append(meta_track)
    for track in mid.tracks:
        mytrack = mido.MidiTrack()
        for message in track:
            if message.is_meta == False:
                if message.type == 'note_on':
                    if(message.velocity == 0):
                        current_time= current_time + message.time
                        new_msg = mido.Message('note_off', note=message.note, velocity=message.velocity, time=current_time)
                        current_time = 0
                        mytrack.append(new_msg)
                    else:
                        current_time= current_time + message.time
                        new_msg = mido.Message('note_on', note=message.note, velocity=message.velocity, time=current_time)
                        current_time = 0
                        mytrack.append(new_msg)
                else:
                    if message.type == 'note_off':
                        current_time= current_time + message.time
                        new_msg = mido.Message('note_off', note=message.note, velocity=message.velocity, time=current_time)
                        current_time = 0
                        mytrack.append(new_msg)
                    else:
                        current_time= current_time + message.time
            else:
                mytrack.append(message)
        my_mid.tracks.append(mytrack)


    my_mid.save(midi_file)
    return notes, length

def process_dataset_midi_files(dataset_path):
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
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

                    if extension == 'midi':
                        process_midi_file(file_path)


if __name__ == "__main__":
    process_dataset_midi_files(DATASET_PATH)

