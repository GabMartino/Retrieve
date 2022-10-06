
import numpy as np
import tensorflow as tf
OutputModelName = "VGG16WithDistractorModel"
modelName = "./VGG16block2FineTuned1024_0"
fromModel = True
dir = "../sketches/png"
dir2 = "../mirflickr25k/mirflickr"
image_size = (224, 224)
batch_size = 32
image_shape = image_size + (3,)

train_ds = tf.keras.utils.image_dataset_from_directory(
    dir,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

train_dsDistractor = tf.keras.utils.image_dataset_from_directory(
    dir2,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

a = np.unique(np.concatenate([y for x, y in train_dsDistractor], axis=0))
b = np.unique(np.concatenate([y for x, y in train_ds], axis=0))
assert a[0] not in b
print(a)
print(b)