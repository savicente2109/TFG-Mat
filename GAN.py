from keras import Sequential
from keras.optimizers import Adam
from discriminator import *
from generator import *
import numpy as np
import matplotlib.pyplot as plt

def create_gan(input_shape, latent_dim):

    discriminator = create_discriminator((32, 32, 3))

    # configuramos el discriminador como NO ENTRENABLE:
    discriminator.trainable = False

    generator = create_generator(latent_dim)

    # modelo secuencial generador + discriminador
    model = Sequential(name="GAN_model")
    model.add(generator)
    model.add(discriminator)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)

    # compilamos el modelo
    # (la red generadora no estaba compilada)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model, generator, discriminator

def generate_real_samples(dataset, n_samples):
    # se eligen elementos aleatoriamente del dataset
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    # se etiquetan como "real" (etiqueta = 1)
    y = np.ones((n_samples, 1))
    return X, y

def save_plot(examples, epoch, n=7):
    # escalamos las imágenes de [-1,1] a [0,1]
    examples = (examples + 1) / 2.0
    # graficamos las imágenes en una cuadrícula n x n
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    # aquí se introduce el nombre del fichero en el que deseamos
    # guardar las imágenes
    filename = '.\generated_images\generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()

# analizar el funcionamiento de la red en términos cuantitativos es complicado,
# y se suelen utilizar técnicas cualitativas: generaremos imágenes de ejemplo tras cada cierto
# número de epochs y al finalizar seleccionaremos los mejores modelos guardados
# (según nuestro criterio humano)
def summarize_performance(epoch, generator, discriminator, dataset, latent_dim, n_samples=150):
    # creamos imágenes reales
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluamos el discriminador en ellas
    _, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)
    # peparamos imágenes falsas
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n_samples)
    # evaluamos el discriminador en ellas
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # resumen del desempeño del discriminador (accuracy)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # GRAFICAMOS LAS IMÁGENES GENERADAS POR G
    save_plot(x_fake, epoch)
    # guardamos el modelo G de la epoch actual (aquí se modifica la ruta)
    filename = '.\generator_models\generator_model_%03d.h5' % (epoch+1)
    generator.save(filename)

# ENTRENAMIENTO COMPLETO DE UNA RED GAN (por defecto 200 epochs y batches de 128 imágenes)
def train_gan(gan, generator, discriminator, dataset, latent_dim, n_epochs=200, n_batch=128):

    # calculamos el número de batches en cada epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # la mitad del batch (half_batch) serán reales y la otra mitad, falsas
    half_batch = int(n_batch / 2)

    # bucle que recorre las epochs
    for i in range(n_epochs):
        
        # bucle que recorre cada batch del dataset en la epoch i
        for j in range(bat_per_epo):

            # ACTUALIZAMOS EL DISCRIMINADOR:

            # seleccionamos imágenes reales aleatoriamente
            # (ya vienen etiquetadas como reales)
            X_real, y_real = generate_real_samples(dataset, half_batch)

            # entrenamos el discriminador con ellas
            # (aquí sí se actualizan sus pesos, recordamos que era NO ENTRENABLE)
            d_loss1, _ = discriminator.train_on_batch(X_real, y_real)

            # generamos imágenes falsas con la red generadora
            # (ya vienen etiquetadas como falsas)
            X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
            
            # entrenamos el discriminador con ellas
            # (aquí sí se actualizan sus pesos, recordamos que era NO ENTRENABLE)
            d_loss2, _ = discriminator.train_on_batch(X_fake, y_fake)

            # AQUÍ TERMINA EL "BUCLE INTERNO" DEL ALGORITMO DE ENTRENAMIENTO
            # CLÁSICO DE LAS GAN: ACTUALIZACIÓN DEL DISCRIMINADOR
            # (con k=1, por eso realmente no es un bucle)

            # AHORA ACTUALIZAMOS EL GENERADOR:

            # generamos puntos del espacio latente (ruido gaussiano)
            X_gan = generate_latent_points(latent_dim, n_batch)

            # creamos etiquetas "invertidas" (llamamos "reales" a las muestras falsas)
            # nuestro objetivo es que sea "bueno" que el discriminador "falle"
            y_gan = np.ones((n_batch, 1))

            # actualizamos el generador con el error del discriminador
            # (entrenamos la GAN COMPLETA, pero el discriminador no verá sus pesos afectados
            # porque es NO ENTRENABLE)
            g_loss = gan.train_on_batch(X_gan, y_gan)

            # resumen del error en este batch (pérdida del discriminador en los datos,
            # pérdida del discriminador en las falsas,
            # pérdida (invertida) del discriminador en las falsas)
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
            (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            
        # llamamos a summarize_performance cada 10 epochs,
        # esto se puede modificar aquí
        if (i+1) % 10 == 0:
            summarize_performance(i, generator, discriminator, dataset, latent_dim)
        # descomentar la línea siguiente (y comentar la anterior)
        # si se desea guardar las imágenes tras cada epoch:
        #summarize_performance(i, generator, discriminator, dataset, latent_dim)