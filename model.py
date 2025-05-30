import tensorflow as tf
from tensorflow.keras import layers

# Constants
BATCH_SIZE = 5
EPOCHS = 7
LEARNING_RATE = 0.0001
IMG_HEIGHT = 28
IMG_WIDTH = 28
Z_DIM = 50

# Generator Model
def build_generator(z_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=z_dim, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(1024, activation="relu"),
        layers.Dense(IMG_HEIGHT * IMG_WIDTH * 1, activation="tanh"),
        layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))
    ])
    return model

# Critic Model
def build_critic():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Wasserstein Loss
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

# Gradient Penalty
def gradient_penalty(critic, real_images, fake_images):
    epsilon = tf.random.uniform([real_images.shape[0], 1, 1, 1], 0.0, 1.0)
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-10)
    penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return penalty
