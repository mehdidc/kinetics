import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

from skimage.transform import resize

class VideoClassifier(object):

    def __init__(self):
        inp = Input((28, 28, 3))
        x = Flatten(name='flatten')(inp)
        x = Dense(100, activation='relu', name='fc1')(x)
        out = Dense(10, activation='softmax', name='predictions')(x)
        self.model = Model(inp, out)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1e-4),
            metrics=['accuracy'])

    def _transform(self, x):
        x = x / 255.
        x = resize(x, (28, 28), preserve_range=True)
        return x

    def fit(self, video_loader):
        nb = len(video_loader)
        X = np.zeros((nb, 28, 28, 3))
        Y = np.zeros((nb, 10))
        for i in range(nb):
            nb_frames = video_loader.nb_frames(i)
            x, y = video_loader.load(i, frame_id=0)
            X[i] = self._transform(x)
            Y[i, y] = 1
        self.model.fit(
            X, Y, 
            batch_size=32, 
            validation_split=0.1, 
            epochs=1
        )

    def predict_proba(self, img_loader):
        nb = len(img_loader)
        X = np.zeros((nb, 28, 28, 3))
        for i in range(nb):
            X[i] = self._transform(img_loader.load(i, frame_id=0))
        return self.model.predict(X)
