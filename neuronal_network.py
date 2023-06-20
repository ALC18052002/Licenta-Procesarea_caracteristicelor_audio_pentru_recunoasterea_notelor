import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = 'dataset/json_test_write.json'
UNIQUE_LABELS_CONFIG_PATH = 'dataset/Unique_Labels_Config/unique_labels_config.json'

def find_index(list_of_pairs, y_uniques):
    for i in range(0 , len(y_uniques)):
        if(len(list_of_pairs) == y_uniques[i].size /2):
            found_index = 1
            for pair in list_of_pairs:
                found_pair = 0
                for arr in y_uniques[i]:
                    if pair[0] == arr[0] and pair[1] == arr[1]:
                        found_pair = 1
                if found_pair == 0:
                    found_index = 0
            if found_index == 1:
                return i

def load_data(data_path, num_mfcc=39, n_fft=2048, hop_length=512):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    n = len(data["labels"])
    total_nr_frames = 0
    for i in range ( 0, n):
        total_nr_frames += len(data["input"]["mfcc"][i])

    mfcc = []
    centroid = np.array(0)
    bandwidth = np.array(0)
    amplitude = np.array(0)
    zcr = np.array(0)
    rms = np.array(0)
    fft = np.array(0)
    chroma = []
    label_sizes = 0
    for i in range(0, n ):
        label_sizes += len(data["labels"][i])
    for i in range ( 0, n):
        amplitude = np.append(amplitude, np.array(data["input"]["amplitude"][i]))
    label_sizes = min(label_sizes, amplitude.size)

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
        
        

    y = [[]]*label_sizes
    label_nr = 0
    for i in range(0,n):
        for label in data["labels"][i]:
            if label_nr < amplitude.size:
                y[label_nr] = []
                for [note, velocity] in label:
                    y[label_nr].append([note, velocity])
            label_nr += 1

    mfcc = np.array(mfcc)
    chroma = np.array(chroma)
    y = np.array(y, dtype=object)

    print(mfcc.shape, centroid.shape, bandwidth.shape, amplitude.shape, zcr.shape, rms.shape, fft.shape, chroma.shape, y.shape)

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
    y = np.array(y)
    return x, y

if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # Create a set to store the unique lists
    y_uniques = set()

    # Convert each list to a frozen set and add it to the unique set
    i = 0 
    for sublist in y:
        i +=1
        frozen_set = []
        for element in sublist:
            frozen_set.append( tuple(element))
        frozen_set = tuple(frozen_set)
        y_uniques.add(frozen_set)

    y_uniques = [np.array(t) for t in y_uniques]

    y_labels = np.zeros(len(y), dtype=int)
    label_nr = 0
    
    unique_labels_config = {
        "index": [],
        "label": []
    }

    index = 0
    for uniques in y_uniques:
        unique_labels_config["index"].append(index)
        unique_labels_config["label"].append(y_uniques[index].tolist())
        index += 1

    with open(UNIQUE_LABELS_CONFIG_PATH, "w") as fp:
        json.dump(unique_labels_config, fp, indent=4)

    for element in y:
        index = find_index(element, y_uniques)
             
        #index = tf.convert_to_tensor(index[0], dtype=tf.int32)
        y_labels[label_nr] = index
        label_nr = label_nr + 1
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2)

    # build network topology
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1],)),

        keras.layers.Dense(16384, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(8192, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(len(y_uniques), activation='softmax')
    ])

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
              loss=keras.losses.sparse_categorical_crossentropy,  # Use categorical_crossentropy for one-hot encoded labels
              metrics=['accuracy'])

    model.summary()

    callback = ModelCheckpoint(
        save_best_only=True,
        save_weights_only=True,
        monitor="accuracy",
        mode="max",
        filepath='neuronal_data_folder/model_weights.hdf5'
    )

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=200, callbacks=[
        callback
    ])

    model.save('neuronal_data_folder/full_model_tf', save_format='tf')

    print(history)
