import mido
import numpy as np
import math
import json
import os

SAMPLE_RATE = 22050
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE
N_FFT = 2048
HOP_LENGTH = 512
FRAME_OFFSET = int( N_FFT / HOP_LENGTH)
DATASET_PATH = 'dataset'
JSON_PATH = 'dataset/midi_to_json.json'

def get_tempo(mid):
    
    for track in mid.tracks:
        for message in track:
            if message.type == 'set_tempo':
                # The tempo value is stored in the "tempo" attribute of the message
                tempo = message.tempo
                return mido.tempo2bpm(tempo)  # Convert tempo to BPM
    
    return None  

def process_midi_file(midi_file):
    mid = mido.MidiFile(midi_file)
    length = mid.length
    ticks_per_beat = mid.ticks_per_beat
    tempo_bpm = get_tempo(mid)
    tempo = mido.bpm2tempo(tempo_bpm)
    current_time = 0
    total_time_ticks = 0


    data = {
        "length":length,
        "ticks_per_beat":ticks_per_beat,
        "tempo": tempo,
        "labels": []
    }

    number_of_samples = int( length * SAMPLE_RATE)
    number_of_frames = int( number_of_samples / HOP_LENGTH)

    raw_labels =  [ [set(), dict()] for _ in range(number_of_frames + 1) ]
    data["labels"] = [[],[]]*(number_of_frames + 2)
    for i in range (0 , number_of_frames + 1):
        data["labels"][i] = []

    notes_with_start_frames = [set(), dict()]

    for track in mid.tracks:
        total_time_ticks = 0
        for message in track:
            if message.is_meta == False:
                if message.type == 'note_on':
                    total_time_ticks = total_time_ticks + message.time
                    event_second = mido.tick2second(tick = total_time_ticks, ticks_per_beat=ticks_per_beat,tempo=tempo)
                    event_sample_nr = int(event_second * SAMPLE_RATE)
                    last_frame_with_note = int(event_sample_nr / HOP_LENGTH)
                    index = int(N_FFT/HOP_LENGTH)
                    start = (last_frame_with_note - index)
                    if message.note in notes_with_start_frames[0]:
                        end_frame = int(event_sample_nr / HOP_LENGTH) - 1

                        start_frame = notes_with_start_frames[1][message.note]["start"]
                        velocity =  notes_with_start_frames[1][message.note]["velocity"]
                        
                        for i in range(start_frame ,end_frame):
                            data["labels"][i].append([message.note, velocity])

                    else:
                        notes_with_start_frames[0].add(message.note)
                        notes_with_start_frames[1][message.note] = {
                            "start": start, 
                            "velocity":message.velocity}
                else:
                    if message.type == 'note_off':
                        total_time_ticks = total_time_ticks + message.time
                        event_second = mido.tick2second(tick = total_time_ticks, ticks_per_beat=ticks_per_beat,tempo=tempo)
                        event_sample_nr = int(event_second * SAMPLE_RATE)
                        end_frame = int(event_sample_nr / HOP_LENGTH) - 1

                        start_frame = notes_with_start_frames[1][message.note]["start"]
                        velocity =  notes_with_start_frames[1][message.note]["velocity"]
                        for i in range(start_frame ,end_frame):
                            data["labels"][i].append([message.note, 64])

                        notes_with_start_frames[0].discard(message.note)
                    else:
                        total_time_ticks = total_time_ticks + message.time
            else:
                total_time_ticks = total_time_ticks + message.time

    return data



def convert_midi_to_Json(dataset_path, json_path):

    json_data={

    }
        
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
                        json_data[file_path] = process_midi_file(file_path)

    with open(json_path, "w") as fp:
        json.dump(json_data, fp, indent=4)


if __name__ == "__main__":
    convert_midi_to_Json(DATASET_PATH, JSON_PATH)