
# model_type = 'ResNet50'
model_type = 'EfficientNetB3'
project_name = 'plant-seedlings-classification'
train_data = 'data/train/'
valid_data = 'data/valid/'
model_save = 'model_save/'
n_class = 12
epochs = 100
batch_size = 16
image_height, image_width = 224, 224
inputshape = (image_height, image_width, 3)
inputsize = (image_height, image_width)
gpus = 1
multi_gpu_flag = False
lr = 1e-3
