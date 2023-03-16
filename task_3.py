import tensorflow as tf
import random
import numpy as np

from sklearn.ensemble import RandomForestClassifier


class CustomSequential(tf.keras.models.Sequential):
    def predict(self, x: np.ndarray):
        """Predict function, convert predictions form onehot vector to integer.

        Parameters
        ----------
        X: np.ndarray
            Predict data.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        return np.argmax(super().predict(x / 255), axis=1)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train function, convert y to onehot structure.

        Parameters
        ----------
        X: np.ndarray
            Train data.
        y: np.ndarray
            Train labels.
        """
        self.fit(X / 255, tf.one_hot(y.astype(np.int32), depth=10))


class CustomRandomForestClassifier(RandomForestClassifier):
    def predict(self, X: np.ndarray):
        """Predict function, reshape X to 1-d numpy array of length 784.

        Parameters
        ----------
        X: np.ndarray
            Predict data.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        return super().predict(X.reshape((X.shape[0], 784)))

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train function, reshape X to 1-d numpy array of length 784.

        Parameters
        ----------
        X: np.ndarray
            Train data.
        y: np.ndarray
            Train labels.
        """
        self.fit(X.reshape((X.shape[0], 784)), y)


class RandomClassifier:
    def __init__(self, random_seed: int = 7, n_epochs: int = 7):
        """Constructor.

        Parameters
        ----------
        random_seed: int
            Seed of the random module.
        n_epochs: int
            Number of epochs in train.
        """
        self.random_seed = random_seed
        self.n_epochs = n_epochs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X_center - transformed data by task condition (10*10 image center).

        Parameters
        ----------
        X: np.ndarray
            Predict data.

        Returns
        -------
        np.ndarray
            Random predictions.

        """
        X_center = X[:, 9:19, 9:19]
        random.seed(self.random_seed)
        return np.array([random.randint(0, 9) for i in range(X.shape[0])])

    def train(self, X: np.ndarray, y: np.ndarray):
        """Pseudo train functionality.

        Parameters
        ----------
        X: np.ndarray
            Train data.
        y: np.ndarray
            Train labels.
        """
        for _ in range(self.n_epochs):
            for _ in zip(X, y):
                pass


def digit_classifier(model_type: str):
    """DigitClassifier function.

    Parameters
    ----------
    model_type: str
        String to specify model.

    Returns
    -------
    np.ndarray
        If dot is valid for next step or not.
    """
    match model_type:
        case 'rf':
            model = CustomRandomForestClassifier(max_leaf_nodes=5,
                                                 max_samples=100)
        case 'cnn':
            model = CustomSequential([
                tf.keras.layers.Conv2D(32, (5, 5), padding='same',
                                       activation='relu',
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (5, 5), padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(strides=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08),
                          loss='categorical_crossentropy', metrics=['acc'])

        case 'rand':
            model = RandomClassifier(15, 10)
        case _:
            raise ValueError

    # Load MNIST dataset from TensorFlow
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data())

    # Added the train method, since some models cannot predict without calling
    # this method. If it is necessary to strictly observe the conditions from
    # doc - train method can be called in the predict method, where necessary,
    # and in all train methods throw Not implemented exception.
    model.train(x_train[:10000], y_train[:10000])
    predictions = model.predict(x_test[:100])
    print(predictions)


if __name__ == '__main__':
    digit_classifier('cnn')


"""!!!!!!!!!!!ATTENTION PLEASE!!!!!!!!!!!
In your task I found one problem.

RandomForestClassifier predict function definition from scikit-learn core:
    def predict(self, X):

tf.keras.models.Sequential predict function definition from Tensorflow core:
    @traceback_utils.filter_traceback
    def predict(
        self,
        x,
        batch_size=None,
        verbose="auto",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):

Because of this there is no opportunity to combine them in one interface. In
my solution I cut all additional parameters from tf method. This isn't correct
solution, but only one from two possible. Second possible solution - add to
Random and RF models additional parameters without functionality.
"""