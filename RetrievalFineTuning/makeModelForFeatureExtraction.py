import datetime

import keras
from keras.applications.vgg16 import VGG16, layers
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.models import *
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras import regularizers


def visualizeFromDataset(ds):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()

def genDataset(dir, image_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dir,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dir,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    return (train_ds, val_ds)

def genModel(input_shape, hiddenNodes1, hiddenNodes2, lastNotTrainableLayer = None):

    ### CREATE PREPROCESSING MODEL

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ]
    )
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    preprocessed_inputs = preprocess_input(x)

    preprocessing_model = tf.keras.Model(inputs, preprocessed_inputs)

    for layer in preprocessing_model.layers:
        layer.trainable = False

    ## ATTACH THE PREPROCESSING MODEL TO THE BASE MODEL
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model(preprocessing_model.output)
    base_model_with_preprocessing = Model(inputs=preprocessing_model.input, outputs=x)

    for layer in base_model.layers:
        print(layer.name)
        if lastNotTrainableLayer != None and layer.name == lastNotTrainableLayer:
            break
        layer.trainable = False
    ## ATTACH THE FF LAYERS TO THE WHOLE MODEL
    x = base_model_with_preprocessing.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(hiddenNodes1, activation=tf.keras.layers.LeakyReLU(alpha=0.3)
              )(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    #x = Dense(hiddenNodes2, activation=tf.keras.layers.LeakyReLU(alpha=0.3)
     #         )(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    predictions = Dense(250, activation="softmax")(x)



    return Model(inputs=base_model_with_preprocessing.input, outputs=predictions)


if __name__ == "__main__":

    modelName = "./VGG16block2FineTuned1024_0"
    fromModel = True
    dir = "../sketches/png"
    image_size = (224, 224)
    batch_size = 32
    image_shape = image_size + (3,)
    hiddenNodes1 = 384
    hiddenNodes2 = 0
    train_ds, val_ds = genDataset(dir, image_size, batch_size)
    model = load_model(modelName)
    if fromModel:
        print("Loading Model")
        lastNotTrainableLayer="block5_pool"
        #model = load_model(modelName)
        check = True
        for layer in model.get_layer("vgg16").layers:
            layer.trainable = not check
            if lastNotTrainableLayer != None and layer.name == lastNotTrainableLayer:
                check = False

        newModel = Model(inputs=model.input, outputs=model.get_layer("batch_normalization").output)
        newModel.add(Dense(hiddenNodes1, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
        newModel.add(Dropout(0.5))
        newModel.add(Dense(250, activation="softmax"))
        newModel.summary()
        exit()
    else:
        model = genModel(image_shape, hiddenNodes1=hiddenNodes1, hiddenNodes2=hiddenNodes2,lastNotTrainableLayer="block4_pool")

    #model.get_layer("dense").kernel_regularizer = None
    #model.get_layer("dense").bias_regularizer = None
    #model.get_layer("dense").activity_regularizer = None
    #model.get_layer("dense_1").kernel_regularizer = None
    #model.get_layer("dense_1").bias_regularizer = None
    #model.get_layer("dense_1").activity_regularizer = None
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    batch_size = 32
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "vgg16block2_"+str(hiddenNodes1)+"_"+ str(hiddenNodes2)
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5',
                                            save_best_only=True
                                               ),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq = 1),
        ]


    start = datetime.datetime.now()

    epochs = 100
    model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        workers=1
    )
    duration = datetime.datetime.now() - start
    model.save("VGG16block5FineTuned" +str(hiddenNodes1)+"_"+str(hiddenNodes2))
    score = model.evaluate(val_ds)
    print("duration: ", duration)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])