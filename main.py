from keras.datasets import cifar10
from GAN import create_gan, train_gan

def load_real_samples():
    # load cifar10 dataset
    (trainX, _), (_, _) = cifar10.load_data()
    # convert from unsigned ints to floats
    X = trainX.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X

# size of the latent space
LATENT_DIM = 100

dataset = load_real_samples()
gan, generator, discriminator = create_gan(input_shape=dataset[0].shape, latent_dim=LATENT_DIM)

# train model
train_gan(gan, generator, discriminator, dataset, LATENT_DIM, n_epochs=1)