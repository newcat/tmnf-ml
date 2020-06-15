import presskey as pk
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from os import listdir
from os.path import isfile, join

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

WIDTH = 200
HEIGHT = 150

ACTIONS = [
    [],
    [pk.LEFT_ARROW],
    [pk.RIGHT_ARROW],
    [pk.UP_ARROW],
    [pk.UP_ARROW, pk.LEFT_ARROW],
    [pk.UP_ARROW, pk.RIGHT_ARROW],
    [pk.DOWN_ARROW],
    [pk.DOWN_ARROW, pk.LEFT_ARROW],
    [pk.DOWN_ARROW, pk.RIGHT_ARROW],
    [pk.UP_ARROW, pk.DOWN_ARROW],
    [pk.UP_ARROW, pk.DOWN_ARROW, pk.LEFT_ARROW],
    [pk.UP_ARROW, pk.DOWN_ARROW, pk.RIGHT_ARROW]
]

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(WIDTH, HEIGHT, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(ACTIONS), activation="sigmoid"))


def load_training_data():
    training_files = [f for f in listdir("training") if isfile(join("training", f))]
    X = None
    y = None
    for f in training_files:
        a = np.load(join("training", f))
        X = np.concatenate((X, a["images"]), axis=0) if X is not None else a["images"]
        y = np.concatenate((y, a["actions"]), axis=0) if y is not None else a["actions"]

    X = X.swapaxes(1, 2)
    # one-hot encoding for action
    y = to_categorical(y, num_classes=len(ACTIONS))

    return train_test_split(X, y, test_size=0.2)


def main():
    X_train, X_test, y_train, y_test = load_training_data()

    y_integers = np.argmax(y_train, axis=1)
    print(np.unique(y_integers))
    class_weights = compute_class_weight('balanced', classes=list(range(len(ACTIONS))), y=y_integers)
    d_class_weights = dict(enumerate(class_weights))
    print(class_weights)

    opt = SGD(lr=1)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, class_weight=d_class_weights)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


main()
