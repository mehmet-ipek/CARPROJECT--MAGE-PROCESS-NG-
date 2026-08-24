import tensorflow_datasets as tfds

dataset = tfds.load(
    "open_images_v4",
    split="train"
)