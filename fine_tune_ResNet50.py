from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.utils import multi_gpu_model
from config import n_class, multi_gpu_flag, gpus

def fine_tune(base_model, method=0):

    if method == 0:
        # base_model.layers.pop()
        # x = base_model.layers[-1].output
        x = base_model.output
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.7)(x)
        predictions = Dense(n_class, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model compiled!')
        if multi_gpu_flag:
            model = multi_gpu_model(model, gpus=gpus)
        return model
    else:
        return base_model
