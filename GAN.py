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

def summarize_performance(epoch, generator, discriminator, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = '.\generator_models\generator_model_%03d.h5' % (epoch+1)
    generator.save(filename)

def train_gan(gan, generator, discriminator, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
    # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = discriminator.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = discriminator.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
            (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # llamamos a summarize_performance cada 10 epochs,
        # esto se puede modificar aquí
        if (i+1) % 10 == 0:
            summarize_performance(i, generator, discriminator, dataset, latent_dim)
        # descomentar la línea siguiente (y comentar la anterior)
        # si se desea guardar las imágenes tras cada epoch:
        #summarize_performance(i, generator, discriminator, dataset, latent_dim)