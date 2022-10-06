import datetime
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.threading.set_intra_op_parallelism_threads(0)

#tf.config.experimental.set_virtual_device_configuration(
 # gpus[0],
 # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
'''
    VGG16 model has 5 blocks
    1) conv, conv, maxpool
    2) conv, conv, maxpool
    3) conv, conv, conv maxpool
    4) conv, conv, conv maxpool
    5) conv, conv, conv maxpool
    6) dense, dense, output

'''
input_size = (224, 224)
batch_size = 32
class CustomVGG16():

    def __init__(self, last_not_trainable_block=5, hidden_layer_size=256, output_size=250):

        ##Set trainable part
        self.model = tf.keras.applications.VGG16(input_shape=input_size + (3,), weights='imagenet',include_top=False)
        trainable = False
        last_non_trainable_layer_name = 'block'+str(last_not_trainable_block)+"_pool"
        for layer in self.model.layers:
            if layer.name == last_non_trainable_layer_name:
                trainable = True
            layer.trainable = trainable

        ## Add FC layers
        x = self.model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(hidden_layer_size, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)  # Dropout layer to reduce overfitting
        predictions = tf.keras.layers.Dense(output_size, activation='softmax')(x)

        self.model = tf.keras.models.Model(inputs=self.model.input, outputs=predictions)


    def printModel(self):
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name, layer.trainable)

    def getModel(self):
        return self.model

def prepareDataset(path):
    ## VGG16 needs a specific preprocessing function
    ## vgg16.preprocess_input will convert the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
    ## Set a bit of data augmentation
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                                                              validation_split=0.2,
                                                              rotation_range=10, # rotation
                                                                width_shift_range=0.2, # horizontal shift
                                                                height_shift_range=0.2, # vertical shift
                                                                zoom_range=0.2, # zoom
                                                                horizontal_flip=True, # horizontal flip
                                                                brightness_range=[0.2,1.2]) # brightness)

    train_ds = datagen.flow_from_directory(
        directory=path,
        target_size=input_size,
        subset="training",
        seed=1337,
        batch_size=batch_size,
        interpolation="bilinear"

    )
    val_ds = datagen.flow_from_directory(
        directory=path,
        target_size=input_size,
        subset="validation",
        seed=1337,
        batch_size=batch_size,
        interpolation="bilinear"
    )

    return (train_ds, val_ds)

def fitModel(model, train_ds, val_ds):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq = 1),
    ]
    # we need to recompile the model for these modifications to take effect
    print("Fit...")

    start = datetime.datetime.now()

    epochs = 50
    model.fit(
        x= train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        workers=1
    )
    duration = datetime.datetime.now() - start
    model.save("modelFineTuned1")
    score = model.evaluate(val_ds)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":

    modelGen = CustomVGG16(last_not_trainable_block=4)
    model = modelGen.getModel()

    dir = '../sketches/png/'

    (train_ds, val_ds) = prepareDataset(dir)

    fitModel(model, train_ds, val_ds)