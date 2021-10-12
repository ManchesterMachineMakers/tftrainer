from argparse import ArgumentParser
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib

# This is the main file.
def main():
    args = ArgumentParser(add_help=True)
    args.add_argument("dir")
    args.add_argument("--outfile", '-o', required=True)
    args.add_argument("--noninteractive", '-y', type=bool)
    args.add_argument("--epochs", '-e', default=3, type=int)
    argv = args.parse_args()
    print("Loading from %s..." % (argv.dir,))
    data_dir = pathlib.Path(argv.dir)
    imgs = list(data_dir.glob("*/*.jpg"))
    image_count = len(imgs)
    print("Found %d images" % (image_count))
    # Assuming the images are all the same size
    first_img = PIL.Image.open(imgs[0])
    img_height, img_width = first_img.size

    batch_size = 32
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    print("~~~~~~~~~DATASET INFO~~~~~~~~~")
    print("Class names:")
    class_names = train_ds.class_names
    for name in class_names:
        print("- %s" % (name,))
    print("Batch size: %d" % (batch_size))
    print("Image dimensions: %d x %d" % (img_height, img_width))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if not argv.noninteractive: input("If that looks right, press Enter to continue. Otherwise, ^C this now.")
    print("Continuing.")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=2)
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names))
    ])
    print("Compiling model")
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    print("Training model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=argv.epochs
    )
