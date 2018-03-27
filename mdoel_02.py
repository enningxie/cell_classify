from keras.applications.resnet50 import ResNet50
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

model.add(ResNet50(include_top='False', weights=None, input_shape=(224, 224, 3)))
# model.add(layers.Flatten())
# add dropout layer
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(128, activation='relu'))
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

train_generator = train_datagen.flow_from_directory(
    train_dir,  # target directory
    target_size=(224, 224),  # resize all images to 150x150
    batch_size=64,
    class_mode='categorical'  # binary
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# fitting the model using a batch generator
history = model.fit_generator(
    train_generator,
    steps_per_epoch=1100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=110
)

print(history.history)