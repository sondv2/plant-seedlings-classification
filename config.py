from fine_tune_xception import fine_tune

# model_type = 'ResNet50'
# model_type = 'EfficientNetB3'
model_type = 'Xception'
project_name = 'plant-seedlings-classification'
train_data = 'data/train/'
valid_data = 'data/valid/'
model_save = 'model_save/'
n_class = 12
epochs = 100
batch_size = 64
image_width, image_height = 240, 240
inputshape = (image_width, image_height, 3)
inputsize = (image_width, image_height)
gpus = 8
multi_gpu_flag = True
lr = 1e-3
