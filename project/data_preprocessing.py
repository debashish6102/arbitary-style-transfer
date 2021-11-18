import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def create_dataset():
    BATCH_SIZE = 2
    BUFFER_SIZE = 100
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    image_size=(256,256)


    train_style_ds = image_dataset_from_directory(
       'shutup/style_images_train/',
        labels=None,
        seed=1337,
        image_size=image_size,
        batch_size=BATCH_SIZE,
    )

    test_style_ds = image_dataset_from_directory(
       'shutup/style_images_test/',
        labels=None,
        seed=1337,
        image_size=image_size,
        batch_size=BATCH_SIZE,
    )

    train_content_ds = image_dataset_from_directory(
       'shutup/content_image_train/',
        labels=None,
        seed=1337,
        image_size=image_size,
        batch_size=BATCH_SIZE,
    )

    test_content_ds = image_dataset_from_directory(
       'shutup/content_image_test/',
        labels=None,
        seed=1337,
        image_size=image_size,
        batch_size=BATCH_SIZE,
    )
    train_dataset = tf.data.Dataset.zip((train_content_ds, train_style_ds))
    test_dataset = tf.data.Dataset.zip((test_content_ds, test_style_ds))
    return train_dataset,test_dataset
