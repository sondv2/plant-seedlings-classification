import efficientnet.keras as efn
from keras import applications
from keras.layers import Dense, BatchNormalization, Activation, Conv2D, MaxPooling2D, Dropout, \
    GlobalAveragePooling2D, MaxPool2D, Flatten
from keras.models import Sequential

def freeze_layers(model, pos=10):
    for layer in model.layers[:pos]:
        layer.trainable = False

def model_define(modeltype, inputshape):

    if modeltype == 'define':
        print('Model: define !')
    elif modeltype == 'EfficientNetB3':
        model = efn.EfficientNetB3(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model)
        print('Model: EfficientNetB3, weights loaded!')
    elif modeltype == 'ResNet50':
        model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=inputshape, pooling='avg')
        freeze_layers(model)
        print('Model: ResNet50, weights loaded!')
    elif modeltype == 'Xception':
        model = applications.Xception(include_top=False, weights='imagenet', input_shape=inputshape)
        # freeze_layers(model)
        model.trainable = False
        print('Model: Xception, weights loaded!')
    else:
        pass

    return model
