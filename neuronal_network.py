import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

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
                y[label_nr][note] = 1
                y[label_nr][note + 128] = velocity
        label_nr += 1

    print(mfcc.shape, centroid.shape, bandwidth.shape, amplitude.shape, zcr.shape, rms.shape, fft.shape, chroma.shape, y.shape)

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

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network topology
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1],)),

        # 1st dense layer
        keras.layers.Dense(2048, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(1024, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(512, activation='relu'),

        # output layer
        keras.layers.Dense(256, activation='softmax')
    ])

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',  # Use categorical_crossentropy for one-hot encoded labels
              metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=1024, epochs=10000)
    print(history)
