from discriminator import create_discriminator

create_discriminator((32,32,3))

model = create_discriminator((32, 32, 3))
# summarize the model
model.summary()