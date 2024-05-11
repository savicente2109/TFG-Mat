from keras import Model
from keras.layers import Input, Dense, Conv2D, LeakyReLU, Dropout, Flatten
from keras.optimizers import Adam

def create_discriminator(input_shape):

    inputs = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="discriminator_model")

    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model