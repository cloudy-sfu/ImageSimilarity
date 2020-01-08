import pickle
import keras
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix


def euclidean_distance(vec):
    x, y = vec
    sum_square = keras.backend.sum(keras.backend.square(x - y), axis=1, keepdims=True)
    return keras.backend.sqrt(keras.backend.maximum(sum_square, keras.backend.epsilon()))

def euclidean_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1

def accuracy(y_true, y_predict):
    return keras.backend.mean(keras.backend.equal(
        y_true, keras.backend.cast(y_predict < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_predict):
    margin = 1
    square_predict = keras.backend.square(y_predict)
    margin_square = keras.backend.square(keras.backend.maximum(margin - y_predict, 0))
    return y_true * square_predict + (1 - y_true) * margin_square

class Siamese:
    def __init__(self):
        self.input_shape = (100, 100, 1)
        self.epochs = 60
        self.batch_size = 128
        self._framework()

    def _framework(self):
        net = keras.models.Sequential()
        net.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape))
        net.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        net.add(keras.layers.MaxPooling2D((2, 2)))
        net.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        net.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        net.add(keras.layers.MaxPooling2D((2, 2)))
        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(128, activation='relu'))
        net.add(keras.layers.Dropout(0.1))
        net.add(keras.layers.Dense(128, activation='relu'))
        net.add(keras.layers.Dropout(0.1))
        net.add(keras.layers.Dense(128, activation='relu'))

        input_1 = keras.layers.Input(shape=self.input_shape)
        input_2 = keras.layers.Input(shape=self.input_shape)
        feature_1 = net(input_1)
        feature_2 = net(input_2)
        distance = keras.layers.Lambda(
            euclidean_distance, output_shape=euclidean_dist_output_shape)([feature_1, feature_2])
        distance = keras.layers.Dense(1, activation='linear')(distance)
        self.model = keras.models.Model([input_1, input_2], distance)

    def load_model(self, filename):
        self.model = keras.models.load_model(filename, custom_objects={
            'keras': keras, 'contrastive_loss': contrastive_loss})

    def train(self, x, y):
        x = np.array(x / 255, dtype=np.uint8)
        rms = keras.optimizers.RMSprop()
        self.model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
        self.model.fit([x[:, 0, :, :, np.newaxis], x[:, 1, :, :, np.newaxis]], y, batch_size=self.batch_size, epochs=self.epochs)
        self.model.save('my_siamese_model.h5')

    def valid(self, x, y):
        y_classify = self.test(x)
        C = confusion_matrix(y_true=y, y_pred=y_classify)
        o1 = PrettyTable(["predict", "", "0", "1"])
        o1.add_row(["truth", "0", C[0, 0], C[0, 1]])
        o1.add_row(["", "1", C[1, 0], C[1, 1]])
        print(o1)

    def test(self, x):
        x = np.array(x / 255, dtype=np.uint8)
        y_ = self.model.predict([x[:, 0, :, :, np.newaxis], x[:, 1, :, :, np.newaxis]])
        return (y_.ravel() < 0.5).astype(int)

if __name__ == "__main__":
    with open('./demo_2.pkl', 'rb') as fp:
        saver = pickle.load(fp)
        X, Y, n = saver['X'], saver['Y'], saver['n']
    train_test_split = int(2 * n * 0.8)
    X_train = X[:train_test_split]
    X_test = X[train_test_split:]
    Y_train = Y[:train_test_split]
    Y_test = Y[train_test_split:]
    siamese = Siamese()
    siamese.train(X_train, Y_train)
    siamese.valid(X_test, Y_test)
    siamese.valid(X_train, Y_train)
