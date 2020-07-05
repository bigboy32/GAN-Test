import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1).astype('float32')
# Normalize the images to [-1, 1]
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 96

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


discriminator = make_discriminator_model()
discriminator_optimizer = tf.optimizers.Adam(lr=1e-4)


def generator_loss(generated_output):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(generated_output), logits=generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

def make_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    model.add(tf.keras.layers.Conv2DTranspose(
        128, (3, 3), padding='same', strides=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(
        1, (3, 3), strides=(2, 2), padding='same'))

    return model


generator = make_generator()
generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4)




def train_step(image):
    fake_image_noise = np.random.randn(BATCH_SIZE, 100).astype('float32')
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(fake_image_noise)
        real_output = discriminator(image)
        fake_output = discriminator(generated_images)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        grad_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        disc_grad = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(grad_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(disc_grad, discriminator.trainable_variables))

        print("Generator Loss : ", np.mean(gen_loss))
        print('Discriminator Loss : ', np.mean(disc_loss), '\n')


fixed_num = np.random.rand(1, 100)

def train(dataset, epochs):
    for epoch in range(epochs):
        plt.imshow(generator(fixed_num).numpy().reshape(28, 28))
        plt.show()
        print("---------------------------Epoch {}/{} -------------------------".format(epoch, epochs), '\n')
        for image in dataset:
            image = tf.cast(image, tf.dtypes.float32)
            train_step(image)

train(train_dataset, 19)

prev_r = None
