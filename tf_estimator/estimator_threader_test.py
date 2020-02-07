import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib.learn import RunConfig
from numpy.testing import assert_almost_equal
from queue import Queue
from threading import Thread


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def load_data(y_name='Species'):
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)


class EstimatorThreader:

    def __init__(self, estimator):
        self.estimator = estimator
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.prediction_thread = Thread(target=self._predict_from_queue, daemon=True)
        self.prediction_thread.start()

    def train(self, input_fn, steps):
        self.estimator.train(input_fn=input_fn, steps=steps)

    def predict(self, features):
        self.input_queue.put(dict(features))
        return self.output_queue.get()   

    def _generate_from_queue(self):
        while True:
            yield self.input_queue.get()

    def _predict_from_queue(self):
        for i in self.estimator.predict(input_fn=self._predict_input_fn):
            self.output_queue.put(i)

    def _predict_input_fn(self):
        dataset = tf.data.Dataset.from_generator(
            self._generate_from_queue,
            output_types={
                'SepalLength': tf.float32,
                'SepalWidth': tf.float32,
                'PetalLength': tf.float32,
                'PetalWidth': tf.float32})
        return dataset


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


def predict_input_fn(features, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    return dataset.batch(batch_size)


def get_unthreaded_estimator(model_path, feature_columns):
    return tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3,
        config=RunConfig().replace(model_dir=model_path),
        model_dir=model_path)


def get_threaded_estimator(model_path, feature_columns):
    return EstimatorThreader(
        get_unthreaded_estimator(model_path, feature_columns))


def get_trained_estimator(get_estimator):
    path = "./models"
    steps = 10
    batch_size = 32
    (train_x, train_y), (test_x, test_y) = load_data() 
    feature_columns = [tf.feature_column.numeric_column(key=k) for k in train_x.keys()]
    m = get_estimator(path, feature_columns)
    m.train(
        input_fn=lambda: train_input_fn(train_x, train_y, batch_size),
        steps=steps)
    return m


def see_if_threaded_faster_than_non_threaded():
    n_epochs = 50
    predict_x = {
        'SepalLength': [5.1],
        'SepalWidth': [3.3],
        'PetalLength': [1.7],   
        'PetalWidth': [0.5]
    }

    m1 = get_trained_estimator(get_estimator=get_unthreaded_estimator)
    m2 = get_trained_estimator(get_estimator=get_threaded_estimator)
    m3 = EstimatorThreader(m1)

    t0 = time.time()
    preds1 = []
    preds2 = []
    preds3 = []
    for _ in range(n_epochs):
        preds1.append(next(m1.predict(input_fn=lambda: predict_input_fn(predict_x, 64))))
    t1 = time.time()
    for _ in range(n_epochs):
        preds2.append(m2.predict(predict_x))
    t2 = time.time()
    for _ in range(n_epochs):
        preds3.append(m3.predict(predict_x))
    t3 = time.time()

    print("unthreaded:", t1-t0)
    print("threaded:", t2-t1)
    print("threaded2:", t3-t2)

    for i in range(len(preds1)):
        assert_almost_equal(preds1[i]["logits"], preds3[i]["logits"])


if __name__ == '__main__':
    see_if_threaded_faster_than_non_threaded()
