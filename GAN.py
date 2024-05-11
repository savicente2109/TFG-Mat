from discriminator import create_discriminator
from generator import create_generator

create_discriminator((32,32,3))

disc = create_discriminator((32, 32, 3))
# summarize the model
disc.summary()

# define the size of the latent space
latent_dim = 100
# define the generator model
gen = create_generator(latent_dim)
# summarize the model
gen.summary()