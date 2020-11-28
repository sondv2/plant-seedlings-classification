from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.optimizers import nadam

def fine_tune(base_model, n_class, multi_gpu_flag=False, gpus=1, method=0):

    if method == 0:
        base_model.layers.pop()
        x = base_model.layers[-1].output
        # x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(100, activation='relu')(x)
        x = BatchNormalization(trainable=True, axis=1)(x)
        x = Dropout(0.5)(x)
        x = Dense(50, activation='relu')(x)
        x = BatchNormalization(trainable=True, axis=1)(x)
        predictions = Dense(n_class, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=nadam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model compiled!')
        if multi_gpu_flag:
            model = multi_gpu_model(model, gpus=gpus)
        return model
    else:
        return base_model
