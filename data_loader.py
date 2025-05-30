import tensorflow as tf
import os

IMG_HEIGHT = 28
IMG_WIDTH = 28

def load_images_from_folder(folder_path, batch_size):
    def process_image(file_path):
        image = tf.io.read_file(file_path)
        file_extension = tf.strings.split(file_path, '.')[-1]
        image = tf.cond(
            tf.equal(tf.strings.lower(file_extension), 'png'),
            lambda: tf.io.decode_png(image, channels=1),
            lambda: tf.io.decode_jpeg(image, channels=1)
        )
        image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image = (tf.cast(image, tf.float32) / 127.5) - 1.0
        return image

    allowed_exts = ['jpg', 'jpeg', 'png']
    all_files = tf.data.Dataset.list_files(os.path.join(folder_path, '*.*'), shuffle=True)

    def is_valid_image(file_path):
        ext = tf.strings.lower(tf.strings.split(file_path, '.')[-1])
        return tf.reduce_any([tf.equal(ext, tf.constant(e)) for e in allowed_exts])

    valid_files = all_files.filter(is_valid_image)
    image_ds = valid_files.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_ds = image_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return image_ds