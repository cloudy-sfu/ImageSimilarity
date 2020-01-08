import keras
import numpy as np
from sklearn.metrics import confusion_matrix


def contrastive_loss(y_true, y_predict):
    margin = 1
    square_predict = keras.backend.square(y_predict)
    margin_square = keras.backend.square(keras.backend.maximum(margin - y_predict, 0))
    return y_true * square_predict + (1 - y_true) * margin_square


class Siamese:
    def __init__(self):
        pass

    def load_model(self, filename):
        self.model = keras.models.load_model(filename, custom_objects={
            'keras': keras, 'contrastive_loss': contrastive_loss})

    def valid(self, x, y):
        y_classify = self.test(x)
        c_mat = confusion_matrix(y_true=y, y_pred=y_classify)
        return c_mat

    def test(self, x):
        x = np.array(x / 255, dtype=np.uint8)
        y_ = self.model.predict([x[:, 0, :, :, np.newaxis], x[:, 1, :, :, np.newaxis]])
        return (y_.ravel() < 0.5).astype(int)
