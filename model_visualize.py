from tensorflow.keras.utils import plot_model


def model_plot(model):
    plot_model(model, to_file='model.png', show_shapes=True)
    # Image(filename='model.png')
