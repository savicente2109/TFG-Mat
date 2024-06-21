# Red Generativa Adversaria (GAN)
 
Este repositorio contiene el código desarrollado como complemento del fundamento teórico de mi Trabajo de Fin de Grado en Matemáticas. Se trata de una red GAN clásica sencilla entrenada con el dataset `CIFAR-10`.

## Contenido

La carpeta `\generator_models` contiene instancias de la red **generadora** cada diez iteraciones de entrenamiento.
La carpeta `\generated_images` contiene muestras de imágenes generadas por cada una de dichas instancias.

El código de la red está distribuido en los ficheros `generator.py`, `discriminator.py` y `GAN.py`. Al ejecutar `main.py`, se entrena la red durante `NUM_EPOCHS` iteraciones. **ATENCIÓN:** Esto sobreescribirá los modelos e imágenes guardados en `\generator_models` y `\generated_images`.

## Referencias

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville y Yoshua Bengio: Generative Adversarial Networks. 2014. https://arxiv.org/pdf/1406.2661.

Jason Brownlee. _Generative Adversarial Networks with Python: Deep Learning Generative Models for Image Synthesis and Image Translation._ Machine Learning Mastery. 2019.
