from keras import Model
from keras.layers import Input, Dense, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Dropout

def create_generator(latent_dim):

    inputs = Input(shape=latent_dim)
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    x = Dense(n_nodes)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, 256))(x)
    # upsample to 8x8
    x = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    # upsample to 16x16
    x = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    # upsample to 32x32
    x = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    # output layer
    outputs = Conv2D(filters=3, kernel_size=(3,3), activation='tanh', padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs, name="generator_model")
    return model