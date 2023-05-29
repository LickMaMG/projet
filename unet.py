import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization, Conv2DTranspose
from keras.optimizers import Adam
from keras.metrics import mean_absolute_error, mean_squared_error

def unet_model(input_shape, filters_init=32, dropout=0.5, activation='relu', output_activation='relu'):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(filters_init, 3, activation=activation,
                   padding='same')(inputs)
    conv1 = Conv2D(filters_init, 3, activation=activation,
                   padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters_init*2, 3, activation=activation,
                   padding='same')(pool1)
    conv2 = Conv2D(filters_init*2, 3, activation=activation,
                   padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters_init*4, 3, activation=activation,
                   padding='same')(pool2)
    conv3 = Conv2D(filters_init*4, 3, activation=activation,
                   padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters_init*8, 3, activation=activation,
                   padding='same')(pool3)
    conv4 = Conv2D(filters_init*8, 3, activation=activation,
                   padding='same')(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters_init*16, 3, activation=activation,
                   padding='same')(pool4)
    conv5 = Conv2D(filters_init*16, 3, activation=activation,
                   padding='same')(conv5)
    drop5 = Dropout(dropout)(conv5)

    # Decoder
    up6 = Conv2DTranspose(filters_init*8, 2, activation=activation,
                          padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(filters_init*8, 3, activation=activation,
                   padding='same')(merge6)
    conv6 = Conv2D(filters_init*8, 3, activation=activation,
                   padding='same')(conv6)

    up7 = Conv2DTranspose(filters_init*4, 2, activation=activation,
                          padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(filters_init*4, 3, activation=activation,
                   padding='same')(merge7)
    conv7 = Conv2D(filters_init*4, 3, activation=activation,
                   padding='same')(conv7)

    up8 = Conv2DTranspose(filters_init*2, 2, activation=activation,
                          padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(filters_init*2, 3, activation=activation,
                   padding='same')(merge8)
    conv8 = Conv2D(filters_init*2, 3, activation=activation,
                   padding='same')(conv8)

    up9 = Conv2DTranspose(filters_init, 2, activation=activation, padding='same')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(filters_init, 3, activation=activation,
                   padding='same')(merge9)
    conv9 = Conv2D(filters_init, 3, activation=activation,
                   padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation=output_activation)(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model



def load_model(
        weights_file_name,
        input_shape=(512,512,1),
        optimizer=Adam(learning_rate=1e-4),
        loss="mse"
        ):
    
    model = unet_model(input_shape=input_shape)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[mean_squared_error, mean_absolute_error])
    model.load_weights(weights_file_name)
    print(model.summary())
    return model
    
# load_model("best_model_weights.h5")