from keras import Model
from keras.layers import Input, Dense, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Dropout
import numpy as np

def create_generator(latent_dim):

    inputs = Input(shape=latent_dim)
	
    # comenzamos con imágenes 4x4
    n_nodes = 256 * 4 * 4
    x = Dense(n_nodes)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, 256))(x)
	
    # 3 capas convolucionales traspuestas para aumentar la dimensionalidad
	# de las imágenes (activación LeakyReLU)

    # aumento a 8x8
    x = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    # aumento a 16x16
    x = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    # aumento a 32x32
    x = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
	
    # capa de salida convolucional con función de activación tangente hiperbólica
	# (el generador genera imágenes con valores de cada canal RGB de cada píxel en [-1, 1])
    outputs = Conv2D(filters=3, kernel_size=(3,3), activation='tanh', padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs, name="generator_model")
	
    # NO COMPILAMOS el modelo, se compilará después la GAN completa

    return model

def generate_latent_points(latent_dim, n_samples):
	# generamos puntos aleatorios del espacio latente
	# (utilizando randn para que sigan una distribución gaussiana)
	x_input = np.random.randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def generate_fake_samples(generator, latent_dim, n_samples):
	# generamos puntos del espacio latente
	x_input = generate_latent_points(latent_dim, n_samples)
	# los "pasamos a través" del generador, es decir,
	# calculamos sus "predicciones" (imágenes falsas)
	X = generator.predict(x_input)
	# les asignamos la etiqueta "falsa" (etiqueta = 0)
	y = np.zeros((n_samples, 1))
	return X, y