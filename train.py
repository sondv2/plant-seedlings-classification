import os
from datetime import datetime

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from data_size import load_data_train_size
from model_define import model_define
from load_data import get_data_size
from model_visualize import model_plot
from load_data import load_datasets, one_hot_encode
from sklearn.model_selection import train_test_split
from checkpoint import callbacks_save_model_weight
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from config import batch_size, epochs, inputshape, valid_data, train_data, project_name, model_type, \
    model_save, n_class, fine_tune, inputsize
from clr_callback import CyclicLR

def train_model(model, steps_per_epoch, nEpochs, trainGenerator,
                valGenerator, validation_steps, resultPath, model_type='VGG16'):

    history = model.fit_generator(generator=trainGenerator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=nEpochs,
                                  validation_data=valGenerator,
                                  validation_steps=validation_steps,
                                  shuffle=True,
                                  verbose=1,
                                  use_multiprocessing=True,
                                  workers=8,
                                  # callbacks=callbacks_save_model_weight()
                                  )

    # Save model
    history_path = os.path.join(resultPath, '%s_%s_modelHistory.npy' % (model_type, project_name))
    # serialize model to JSON
    model_json_path = os.path.join(resultPath, '%s_%s_model_json.json' % (model_type, project_name))
    model_path = os.path.join(resultPath, '%s_%s_modelArchitecture.h5' % (model_type, project_name))
    weights_path = os.path.join(resultPath, '%s_%s_modelWeights.h5' % (model_type, project_name))

    np.save(history_path, history.history)
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
        json_file.close()
    model.save(model_path)
    model.save_weights(weights_path)
    return history, model


# fix random seed for reproducibility
seed = 2020
np.random.seed(seed)

if __name__ == "__main__":
    # pretrained
    model_base = model_define(modeltype=model_type, inputshape=inputshape)
    # print(model_base.summary())
    # fully connected layers for learning weights (fine-tune)
    modelUntrained = fine_tune(model_base, n_class)

    print(modelUntrained.summary())

    # load data
    X, y, num_classes = load_datasets(train_data)

    # split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=seed)

    print('Train Shape: {}'.format(X_train.shape))
    print('Valid Shape: {}'.format(X_val.shape))

    # Data Preprocessing and rescaling
    train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                        # featurewise_center=False,  # set input mean to 0 over the dataset
                                        # samplewise_center=False,  # set each sample mean to 0
                                        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                        # samplewise_std_normalization=False,  # divide each input by its std
                                        # zca_whitening=False,  # apply ZCA whitening
                                        # rotation_range=30,
                                        # width_shift_range=0.2,
                                        # height_shift_range=0.2,
                                        # horizontal_flip=True,
                                        # shear_range=0.2,
                                        # zoom_range=0.2,
                                        # fill_mode='nearest'
                                        rotation_range=20,
                                        zoom_range=0.15,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.15,
                                        horizontal_flip=True,
                                        brightness_range=[0.4, 1],
                                        )
    validation_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_data_gen.flow(np.array(X_train), y_train, batch_size=batch_size)
    # validation_generator = train_data_gen.flow(np.array(X_train), y_train, batch_size=batch_size)
    validation_generator = validation_data_gen.flow(np.array(X_val), y_val, batch_size=batch_size)

    # train_generator = train_data_gen.flow_from_directory(train_data, target_size=inputsize,
    #                                                      batch_size=batch_size, class_mode='categorical')
    #
    # validation_generator = validation_data_gen.flow_from_directory(valid_data, target_size=inputsize,
    #                                                                batch_size=batch_size, class_mode='categorical')
    steps_per_epoch = len(y_train) // batch_size
    validation_steps = len(y_val) // batch_size

    # model_plot(modelUntrained)

    history, modelTrained = train_model(model=modelUntrained,
                                        steps_per_epoch=steps_per_epoch,
                                        nEpochs=epochs,
                                        trainGenerator=train_generator,
                                        valGenerator=validation_generator,
                                        validation_steps=validation_steps,
                                        resultPath=model_save,
                                        model_type=model_type)

    # evaluate(modelTrained, validation_generator, train_generator, batch_size)
