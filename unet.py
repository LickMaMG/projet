import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization, Conv2DTranspose


def unet_model(input_shape, filters_init=32):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(filters_init, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(filters_init, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters_init*2, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters_init*2, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters_init*4, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters_init*4, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters_init*8, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters_init*8, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters_init*16, 3, activation='relu',
                   padding='same')(pool4)
    conv5 = Conv2D(filters_init*16, 3, activation='relu',
                   padding='same')(conv5)
    drop5 = Dropout(0.2)(conv5)

    # Decoder
    up6 = Conv2D(filters_init*8, 2, activation='relu',
                 padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(filters_init*8, 3, activation='relu',
                   padding='same')(merge6)
    conv6 = Conv2D(filters_init*8, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(filters_init*4, 2, activation='relu',
                 padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(filters_init*4, 3, activation='relu',
                   padding='same')(merge7)
    conv7 = Conv2D(filters_init*4, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(filters_init*2, 2, activation='relu',
                 padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(filters_init*2, 3, activation='relu',
                   padding='same')(merge8)
    conv8 = Conv2D(filters_init*2, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(filters_init, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(filters_init, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(filters_init, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='relu')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model


def unet_model2(input_shape, filters_init=32):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(filters_init, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(filters_init, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters_init*2, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters_init*2, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters_init*4, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters_init*4, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters_init*8, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters_init*8, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters_init*16, 3, activation='relu',
                   padding='same')(pool4)
    conv5 = Conv2D(filters_init*16, 3, activation='relu',
                   padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = Conv2DTranspose(filters_init*8, 2, activation='relu',
                          padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(filters_init*8, 3, activation='relu',
                   padding='same')(merge6)
    conv6 = Conv2D(filters_init*8, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(filters_init*4, 2, activation='relu',
                          padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(filters_init*4, 3, activation='relu',
                   padding='same')(merge7)
    conv7 = Conv2D(filters_init*4, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(filters_init*2, 2, activation='relu',
                          padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(filters_init*2, 3, activation='relu',
                   padding='same')(merge8)
    conv8 = Conv2D(filters_init*2, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(filters_init, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(filters_init, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(filters_init, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='relu')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model


def unet_model3(input_shape, use_batch_norm, activation, output_activation):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation=activation, padding='same')(inputs)
    if use_batch_norm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation=activation, padding='same')(conv1)
    if use_batch_norm:
        conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation=activation, padding='same')(pool1)
    if use_batch_norm:
        conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation=activation, padding='same')(conv2)
    if use_batch_norm:
        conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation=activation, padding='same')(pool2)
    if use_batch_norm:
        conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation=activation, padding='same')(conv3)
    if use_batch_norm:
        conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation=activation, padding='same')(pool3)
    if use_batch_norm:
        conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation=activation, padding='same',)(conv4)
    if use_batch_norm:
        conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation=activation, padding='same')(pool4)
    if use_batch_norm:
        conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation=activation, padding='same')(conv5)
    if use_batch_norm:
        conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = Conv2D(512, 2, activation=activation, padding='same')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation=activation, padding='same')(merge6)
    if use_batch_norm:
        conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation=activation, padding='same')(conv6)
    if use_batch_norm:
        conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation=activation, padding='same')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation=activation, padding='same')(merge7)
    if use_batch_norm:
        conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation=activation, padding='same')(conv7)
    if use_batch_norm:
        conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation=activation, padding='same')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation=activation, padding='same')(merge8)
    if use_batch_norm:
        conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation=activation, padding='same')(conv8)
    if use_batch_norm:
        conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation=activation, padding='same')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation=activation, padding='same')(merge9)
    if use_batch_norm:
        conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation=activation, padding='same')(conv9)
    if use_batch_norm:
        conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation=output_activation)(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model
