import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.utils import to_categorical
from keras import metrics
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Maximum
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate, add
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
import os
import nibabel as nib
import config
from tqdm import tqdm
import Model

input_img = Input((128, 128, 128, 4))
model = Model.Unet_3d(input_img, 8, 0.1, True)
learning_rate = 0.001
epochs = 60
decay_rate = 0.0000001
model.compile(optimizer=Adam(lr=learning_rate, decay=decay_rate), loss=Model.dice_coef_loss, metrics=[Model.dice_coef])
model.summary()

ROOT = "BrainTumourData"
path = ROOT
all_images = sorted(os.listdir(config.IMAGES_DATA_DIR))
all_masks = sorted(os.listdir(config.LABELS_DATA_DIR))
image_data = np.zeros((240, 240, 155, 4))
mask_data = np.zeros((240, 240, 155))

x = all_images[1]
image_path = os.path.join(config.IMAGES_DATA_DIR, x)

for epoch in tqdm(range(epochs), desc="epochs:"):
    for num in tqdm(range(180), desc="data:"):
        # data preprocessing starts here
        x = all_images[num]
        # print(x)
        image_path = os.path.join(config.IMAGES_DATA_DIR, x)
        # images = os.listdir(image_path)
        # images.sort()
        img = nib.load(image_path)
        image_data = img.dataobj
        image_data = np.asarray(image_data)

        y = all_masks[num]
        # print(y)
        mask_path = os.path.join(config.LABELS_DATA_DIR, y)
        # masks = os.listdir(masks_path)
        # masks.sort()
        msk = nib.load(mask_path)
        mask_data = msk.dataobj
        mask_data = np.asarray(mask_data)
        print("Entered ground truth")

        reshaped_image_data = image_data[56:184, 80:208, 13:141, :]

        reshaped_mask_data = mask_data[56:184, 80:208, 13:141]

        reshaped_image_data = reshaped_image_data.reshape(1, 128, 128, 128, 4)
        reshaped_mask_data = reshaped_mask_data.reshape(1, 128, 128, 128)
        reshaped_mask_data[reshaped_mask_data == 4] = 3
        reshaped_mask_data_flatten = reshaped_mask_data.flatten()
        # y_to = keras.utils.to_categorical(y_to,num_classes=2)
        print(reshaped_mask_data.shape)
        print("Number of classes", np.unique(reshaped_mask_data_flatten))

        reshaped_mask_data = to_categorical(reshaped_mask_data, num_classes=4)

        print(reshaped_image_data.shape)
        print(reshaped_mask_data.shape)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('braintumor_model.h5', verbose=1, save_best_only=True)
        model.fit(x=reshaped_image_data, y=reshaped_mask_data, epochs=1, callbacks=[checkpoint_cb])
        model.save('3d_braintumor_model.h5')

model.save('3d_braintumor_model.h5')


