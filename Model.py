import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D, MaxPooling3D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import nibabel as nib


def conv_block(input_mat, num_filters, kernel_size, batch_norm):
    X = Conv3D(num_filters, kernel_size=(kernel_size, kernel_size, kernel_size), strides=(1, 1, 1), padding='same')(
        input_mat)
    if batch_norm:
        X = BatchNormalization()(X)

    X = Activation('relu')(X)

    X = Conv3D(num_filters, kernel_size=(kernel_size, kernel_size, kernel_size), strides=(1, 1, 1), padding='same')(X)
    if batch_norm:
        X = BatchNormalization()(X)

    X = Activation('relu')(X)

    return X


def Unet_3d(input_img, n_filters=8, dropout=0.2, batch_norm=True):
    c1 = conv_block(input_img, n_filters, 3, batch_norm)
    p1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv_block(p1, n_filters * 2, 3, batch_norm)
    p2 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv_block(p2, n_filters * 4, 3, batch_norm)
    p3 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv_block(p3, n_filters * 8, 3, batch_norm)
    p4 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv_block(p4, n_filters * 16, 3, batch_norm)

    u6 = Conv3DTranspose(n_filters * 8, (3, 3, 3), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, n_filters * 8, 3, batch_norm)
    c6 = Dropout(dropout)(c6)
    u7 = Conv3DTranspose(n_filters * 4, (3, 3, 3), strides=(2, 2, 2), padding='same')(c6)

    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, n_filters * 4, 3, batch_norm)
    c7 = Dropout(dropout)(c7)
    u8 = Conv3DTranspose(n_filters * 2, (3, 3, 3), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])

    c8 = conv_block(u8, n_filters * 2, 3, batch_norm)
    c8 = Dropout(dropout)(c8)
    u9 = Conv3DTranspose(n_filters, (3, 3, 3), strides=(2, 2, 2), padding='same')(c8)

    u9 = concatenate([u9, c1])

    c9 = conv_block(u9, n_filters, 3, batch_norm)
    outputs = Conv3D(4, (1, 1, 1), activation='softmax')(c9)
    print("!!!!!!!!!!!!!!!!!!!")
    print(outputs.shape)
    model = Model(inputs=input_img, outputs=outputs)

    return model


def standardize(image):
    standardized_image = np.zeros(image.shape)
    # iterate over the `z` dimension
    for z in range(image.shape[2]):
        # get a slice of the image
        # at channel c and z-th dimension `z`
        image_slice = image[:, :, z]

        # subtract the mean from image_slice
        centered = image_slice - np.mean(image_slice)

        # divide by the standard deviation (only if it is different from zero)
        if (np.std(centered) != 0):
            centered = centered / np.std(centered)

            # update  the slice of standardized image
        # with the scaled centered and scaled image
        standardized_image[:, :, z] = centered

    ### END CODE HERE ###

    return standardized_image


def dice_coef(y_true, y_pred, epsilon=0.00001):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf

    """
    axis = (0, 1, 2, 3)
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true * y_true, axis=axis) + K.sum(y_pred * y_pred, axis=axis) + epsilon
    return K.mean((dice_numerator) / (dice_denominator))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


class DiceCoefficientLoss(keras.losses.Loss):
    def __init__(self, epsilon=0.00001, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def dice_coef(self, y_true, y_pred):
        axis = (0, 1, 2, 3)
        dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + self.epsilon
        dice_denominator = K.sum(y_true * y_true, axis=axis) + K.sum(y_pred * y_pred, axis=axis) + self.epsilon
        return K.mean((dice_numerator) / (dice_denominator))

    def call(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "epsilon": self.epsilon}
