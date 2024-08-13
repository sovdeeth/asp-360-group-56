from format_data import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import Sequential
from keras import layers
import keras
import matplotlib.pyplot as plt


def get_data(segment_length, features):
    data = read_data_np(segment_length)
    normalizer = DataNormalizer(data, segment_length, features)
    normed_data = normalizer.normalize(data)
    inputs, labels = split_data(normed_data, segment_length, features)
    # Split the data into training and testing sets
    inputs_train_val, inputs_test, labels_train_val, labels_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
    inputs_train, inputs_val, labels_train, labels_val = train_test_split(inputs_train_val, labels_train_val, test_size=0.25, random_state=42)

    return (inputs_train, labels_train), (inputs_val, labels_val), (inputs_test, labels_test), normalizer

normalizer = None

# print("inputs_train shape:", train[0].shape)  # Should be (number of samples, segment_length, 4)
# print(train[0][:10])
# print("inputs_valid shape:", val[0].shape)  # Should be (number of samples, segment_length)
# print("inputs_test shape:", test[0].shape)  # Should be (number of samples, segment_length, 4)
# print("labels_train shape:", train[1].shape)  # Should be (number of samples, 3)
# print(train[1][:1])
# print("labels_valid shape:", val[1].shape)  # Should be (number of samples, 3)
# print("labels_test shape:", test[1].shape)  # Should be (number of samples, 3)

def lat_deviation(y_true, y_pred):
    return tf.abs(normalizer.unnormalize_output(y_pred[-1], 0) - normalizer.unnormalize_output(y_true[-1], 0))

def long_deviation(y_true, y_pred):
    return tf.abs(normalizer.unnormalize_output(y_pred[-1], 1) - normalizer.unnormalize_output(y_true[-1], 1))

def wind_deviation(y_true, y_pred):
    return tf.abs(normalizer.unnormalize_output(y_pred[-1], 2) - normalizer.unnormalize_output(y_true[-1], 2))



def create_model(segment_length, input_features, output_features = 3):
    # Build the model
    model = Sequential()

    # First LSTM layer with 50 units
    model.add(layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(segment_length, input_features)))

    # Second LSTM layer with 50 units
    model.add(layers.LSTM(50, activation='relu'))

    # First Dense layer with 25 units
    model.add(layers.Dense(10, activation='relu'))

    # First Dense layer with 25 units
    model.add(layers.Dense(5, activation='relu'))

    # Second Dense layer with 3 units (output layer)
    model.add(layers.Dense(output_features))

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=[lat_deviation, long_deviation, wind_deviation])

    # Summary of the model
    model.summary()

    return model

def train(iteration, model, train, valid, batch_size=32, epochs=50):
    global segments
    checkpoint_filepath = './output/checkpoints/Model_{}_S{}_(B{}-E{}).keras'.format(iteration, segments, batch_size, "{epoch:02d}")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_long_deviation',
        mode='min',
        save_best_only=True)

    history = model.fit(train[0], train[1], batch_size, epochs, validation_data=(valid[0], valid[1]), callbacks=model_checkpoint_callback)
    model.save('./output/Model Trials/Model_{}_S{}_(B{}-E{}).keras'.format(iteration, segments, batch_size, epochs))

    plt.plot(history.history['loss'][2:])
    plt.plot(history.history['val_loss'][2:])
    plt.title('model loss')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./output/Model Trials/(Cat) Training_Curve-Model_{}_S{}_(B{}-E{})'.format(iteration, segments, batch_size, epochs))
    plt.show()

    plt.plot(history.history['lat_deviation'])
    plt.plot(history.history['long_deviation'])
    plt.plot(history.history['wind_deviation'])
    plt.plot(history.history['val_lat_deviation'])
    plt.plot(history.history['val_long_deviation'])
    plt.plot(history.history['val_wind_deviation'])
    plt.title('model deviation')
    plt.ylabel('deviation (degrees)')
    plt.xlabel('epoch')
    plt.legend(['train_lat', 'train_long', 'train_wind', 'val_lat', 'val_long', 'val_wind'], loc='upper left')
    plt.savefig('./output/Model Trials/(Cat) Deviation_Curve-Model_{}_S{}_(B{}-E{})'.format(iteration, segments, batch_size, epochs))
    plt.show()

def model_eval(model,test,iteration, batch_size, epochs):
    test_eval = model.evaluate(test[0], test[1])
    f = open("test_eval.txt", "a")
    f.write("(Cat) Iteration {} (S{}, B{}, E{}): ".format(str(iteration), str(segments), str(batch_size), str(epochs)) + str(test_eval) + "\n")
    f.close()

    #open and read the file after the appending:
    f = open("test_eval.txt", "r")
    print(f.read())


iteration = 4

segments = 3
features = 10

epochs = 75
batch_size = 32

if __name__ == "__main__":
    train_set, valid_set, test_set, normalizer = get_data(segments, features)
    model = create_model(segments, features)
    train(iteration, model, train_set, valid_set, batch_size, epochs)
    model_eval(model, test_set, iteration, batch_size, epochs)