import tensorflow as tf
from tensorflow import keras
import math
import glob
import os


# class customDatagen(keras.utils.Sequence):

#     def __init__(self, x_set, y_set, batch_size):
#         self.x, self.y = x_set, y_set
#         self.batch_size = batch_size


#     def __len__(self):
#         return math.ceil(len(self.x)/self.batch_size)


#     def __getitem__(self, idx):
#         low = idx*self.batch_size
#         high = min(low + self.batch_size, len(self.x))
#         batch_x = self.x[low:high]
#         batch_y = self.y[low:high]
#         return batch_x, batch_y


def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (128, 128))
    image = tf.image.per_image_standardization(image)  # You can adjust preprocessing as needed
    return image

def create_shifted_frames(directory, image_type = 'jpg'):
    image_files = glob.glob(os.path.join(directory, '*.'+image_type))
    print("image files ready")
    train_ratio = 0.7

    train_images = image_files[0:int(train_ratio*len(image_files))]
    validation_images = image_files[int(train_ratio*len(image_files)):]
    print("Train valid split DONE")
    train_images_x = train_images[:len(train_images)-1]
    train_images_y = train_images[3:]
    train_x = tf.data.Dataset.from_tensor_slices(train_images_x)
    train_x = train_x.map(load_and_preprocess_image)
    train_y = tf.data.Dataset.from_tensor_slices(train_images_y)
    train_y = train_y.map(load_and_preprocess_image)

    valid_images_x = validation_images[:len(validation_images)-1]
    valid_images_y = validation_images[3:]
    valid_x = tf.data.Dataset.from_tensor_slices(valid_images_x)
    valid_x = valid_x.map(load_and_preprocess_image)
    valid_y = tf.data.Dataset.from_tensor_slices(valid_images_y)
    valid_y = valid_y.map(load_and_preprocess_image)

    def create_sequence(images):
        return tf.reshape(images, (3, 128, 128, 3))
    
    train_x = train_x.window(size=3, shift=1, drop_remainder=True)
    train_x = train_x.flat_map(lambda x: x.batch(3))
    train_x = train_x.map(create_sequence)

    valid_x = valid_x.window(size=3, shift=1, drop_remainder=True)
    valid_x = valid_x.flat_map(lambda x: x.batch(3))
    valid_x = valid_x.map(create_sequence)

    train_dataset = tf.data.Dataset.zip((train_x, train_y))
    valid_dataset = tf.data.Dataset.zip((valid_x, valid_y))
    return train_dataset, valid_dataset

if __name__ == '__main__':
    train_x, train_y, valid_x, valid_y = create_shifted_frames('Images_jet_2010')
    
