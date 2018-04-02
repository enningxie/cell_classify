# a small convnet for dog vs cat classification
from keras import layers
from keras import models
from keras import optimizers, losses, metrics
from keras.preprocessing.image import ImageDataGenerator
# from keras_05_02 import train_dir, validation_dir
import matplotlib.pyplot as plt
train_dir = '/home/enningxie/Documents/DataSets/CELL_IMAGES/CELL_images/train'
validation_dir = '/home/enningxie/Documents/DataSets/CELL_IMAGES/CELL_images/test'

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.7))
model.add(layers.Dense(3, activation='softmax'))

print(model.summary())

# compile_op
model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adam(lr=0.001),
              metrics=[metrics.categorical_accuracy])

# preprocessing the imagedata
# rescales all images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,  # target directory
    target_size=(50, 50),  # resize all images to 150x150
    batch_size=32,
    class_mode='categorical'  # binary
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)

# fitting the model using a batch generator
history = model.fit_generator(
    train_datagen_aug,
    steps_per_epoch=150,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)

print(history.history)