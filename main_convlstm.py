import numpy as np
import os, sys
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import layers
from custom_datagen import create_shifted_frames
from convLSTM import LSTM_pred
#physical_devices = tf.config.list_physical_devices()

# Set TensorFlow to use only CPU
#tf.config.set_visible_devices(physical_devices[0], 'CPU')

folder = 'Images_jet_2010'

# Original Dimensions
# image_width = 128
# image_height = 128
# channels = 3

train, valid = create_shifted_frames(folder)




# def create_shifted_frames(data):
#     x = data[0 : data.shape[0] - 1, :, :, :]
#     y = data[1 : data.shape[0], :, :, :]
#     return x, y

#x, y = create_shifted_frames(dataset)

# indexes = np.arange(dataset.shape[0])

# train_index = indexes[: int(0.9 * dataset.shape[0])]
# val_index = indexes[int(0.9 * dataset.shape[0]):]
# train_dataset = dataset[train_index]
# val_dataset = dataset[val_index]



# Define modifiable training hyperparameters.
epochs = 1
batch_size = 4

if os.path.exists('model.h5'):
    model = keras.models.load_model('model.h5')

else:
    model = LSTM_pred()

model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(),
)

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

print("Ready to train")


# Fit the model to the training data.
model.fit(
    train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=valid,
    callbacks=[early_stopping, reduce_lr],
)

model.save("params.h5")