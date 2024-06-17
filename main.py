from keras.datasets import cifar10
from GAN import create_gan, train_gan

def load_real_samples():
    # cargamos el dataset (obviamos las etiquetas porque no nos
    # ocupa la tarea de clasificar las imágenes como animales, vehículos, etc.)
    (trainX, _), (_, _) = cifar10.load_data()
    # convertimos los valores de los píxeles a tipo real (float)
    X = trainX.astype('float32')
    # y los escalamos de [0, 255] a [-1, 1]
    X = (X - 127.5) / 127.5
    return X

# tamaño del espacio latente para la red generadora
LATENT_DIM = 100

# número de episodios de entrenamiento
NUM_EPOCHS = 200

# cargamos el dataset real
dataset = load_real_samples()

# creamos la GAN
gan, generator, discriminator = create_gan(input_shape=dataset[0].shape, latent_dim=LATENT_DIM)

# entrenamos el modelo
train_gan(gan, generator, discriminator, dataset, LATENT_DIM, n_epochs=NUM_EPOCHS)