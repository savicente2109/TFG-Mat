from keras import Model
from keras.layers import Input, Dense, Conv2D, LeakyReLU, Dropout, Flatten
from keras.optimizers import Adam

def create_discriminator(input_shape):

    inputs = Input(shape=input_shape)

    # 4 capas convolucionales con activación LeakyReLU

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    # reescalamos la salida bidimensional para utilizarla como entrada
    # de la última capa densa

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    # capa densa con una única salida (clasificación binaria)
    # con función de activación sigmoide (logística)

    outputs = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="discriminator_model")

    # compilamos el modelo
    # se puede probar el algoritmo Adam con otras configuraciones de parámetros
    # u otros algoritmos
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # por supuesto, utilizamos la entropía cruzada binaria como función de pérdida
    # del discriminador (clasificador binario)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model