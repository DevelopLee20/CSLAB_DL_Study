import image_to_dataset as itd

X_train, y_train = itd.image_to_input_data(
    dir_path='./data',
    grayscale=True,
    processing=0,
    validation=0.0
)

print(X_train, len(X_train), y_train, len(y_train))