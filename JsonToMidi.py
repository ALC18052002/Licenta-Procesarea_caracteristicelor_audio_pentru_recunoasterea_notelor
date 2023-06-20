import mido
import json

HOP_LENGTH = 512
N_FFT = 2048
SAMPLE_RATE = 22050
FRAME_OFFSET = int(N_FFT/HOP_LENGTH)
JSON_PATH = 'dataset/predictions.json'

def recreate_midi_file(json_path):
    # Load the note data from the JSON file
    with open(json_path, "r") as fp:
        data = json.load(fp)

    labels = data["labels"]

    my_mid = mido.MidiFile()


    meta_track = mido.MidiTrack()

    ticks_per_beat = data["ticks_per_beat"]
    current_time = 0
    tempo = mido.bpm2tempo(120)
    tempo_msg = mido.MetaMessage('set_tempo', tempo=tempo)
    meta_track.append(tempo_msg)
    my_mid.ticks_per_beat = ticks_per_beat
    my_mid.tracks.append(meta_track)

    note_track = mido.MidiTrack()

    midi_events = []

    notes = set()
    frame_nr = 0
    last_midi_message_time = 0
    for frame in labels:
        notes_to_discard = set()
        for set_note in notes:
            exists = 0
            for [note, velocity] in frame:
                if set_note == note:
                    exists = 1
            if exists == 0:
                current_time_seconds = (frame_nr * HOP_LENGTH )/ SAMPLE_RATE
                midi_events.append({"note": set_note, "message":'note_off', "velocity":int(velocity), "time_seconds":current_time_seconds})
                notes_to_discard.add(set_note)
        for note_discard in notes_to_discard:
            notes.discard(note_discard)
        note_discard = set()
        for [note, velocity] in frame:
            if note not in notes:
                current_time_seconds = (frame_nr + FRAME_OFFSET )  * HOP_LENGTH / SAMPLE_RATE
                midi_events.append({"note": note, "message":'note_on', "velocity":int(velocity), "time_seconds":current_time_seconds})
                notes.add(note)
        frame_nr += 1
    

    sorted_list = sorted(midi_events, key=lambda x: x["time_seconds"])
    last_midi_message_time = 0
    for event in sorted_list:
        current_time_seconds = event["time_seconds"]
        offset_time = current_time_seconds - last_midi_message_time
        last_midi_message_time = current_time_seconds
        offset_ticks = int(mido.second2tick(second=offset_time,ticks_per_beat=ticks_per_beat, tempo= tempo))
        new_msg = mido.Message(event["message"], note=event["note"], velocity=event["velocity"], time=offset_ticks)
        note_track.append(new_msg)

    my_mid.tracks.append(note_track)

    my_mid.save('reconstructed_Midi.mid')    
    # Create a new MIDI file

        
if __name__ == "__main__":
    # Provide the paths for the JSON file and the output MIDI file

    # Call the function to recreate the MIDI file
    recreate_midi_file(JSON_PATH)

