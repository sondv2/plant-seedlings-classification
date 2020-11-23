
def save_data_train_size(num_samples, num_train):
    with open('data/config.txt', 'w') as file:
        file.write(str(num_samples) + "\n")
        file.write(str(num_train) + "\n")
        file.write(str(num_samples - num_train) + "\n")
        file.close()
def load_data_train_size():
    with open('data/config.txt', 'r') as file:
        num_samples = int(file.readline())
        num_train = int(file.readline())
        num_val = int(file.readline())
        file.close()
    return num_train, num_val
