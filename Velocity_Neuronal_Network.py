import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "/Users/alexanlazar/Licenta-Procesarea_caracteristicelor_audio_pentru_recunoasterea_notelor/dataset/json_test_write.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    mfcc = np.array(data["input"]["mfcc"][0])
    centroid = np.array(data["input"]["centroid"][0])
    bandwidth = np.array(data["input"]["bandwidth"][0])
    amplitude = np.array(data["input"]["amplitude"][0])
    zcr = np.array(data["input"]["zcr"][0])
    rms = np.array(data["input"]["rms"][0])
    fft = np.array(data["input"]["fft"][0])
    chroma = np.array(data["input"]["chroma"][0])
    y = np.zeros((amplitude.size, 256))

    label_nr = 0
    for label in data["labels"][0]:
        for [note, velocity] in label:
            if label_nr < amplitude.size:
                y[label_nr][note] = velocity
        label_nr += 1

    x = np.empty((amplitude.size, 5 + mfcc[0].size + 12))
    for i in range(0, amplitude.size ):
        x[i][0] = centroid[i]
        x[i][1] = bandwidth[i]
        x[i][2] = amplitude[i]
        x[i][3] = zcr[i]
        x[i][4] = rms[i]
        x[i][5] = fft[i]
        for j in range(0, mfcc[0].size):
            x[i][j+5] = mfcc[i][j]
        
        for j in range(0, 12):
            x[i][j+5+mfcc[0].size] = chroma[j][i]

    return x, y

if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # Create a set to store the unique lists
    y_uniques = set()

    # Convert each list to a frozen set and add it to the unique set
    for sublist in y:
        frozen_set = tuple(sublist)
        y_uniques.add(frozen_set)

    y_uniques = [np.array(t) for t in y_uniques]
    
    y_labels = np.zeros(len(y), dtype=int)
    label_nr = 0
    for element in y:
        index = np.where((y_uniques == element).all(axis=1))[0]
        #index = tf.convert_to_tensor(index[0], dtype=tf.int32)
        y_labels[label_nr] = index[0]
        label_nr = label_nr + 1
    print(y_labels.size)
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2)

    # build network topology
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1],)),

        # 1st dense layer
        keras.layers.Dense(4112, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(2056, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(1024, activation='relu'),

        keras.layers.Dense(512, activation='relu'),

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
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=100, callbacks=[
        callback
    ])

    model.save('neuronal_data_folder/full_model_tf', save_format='tf')

    print(history)
