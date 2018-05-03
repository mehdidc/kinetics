import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

train_size = 1000
test_size = 450
labels = [
    'catching or throwing baseball',
    'catching or throwing frisbee',
    'catching or throwing softball',
    'passing American football (in game)',
    'dribbling basketball',
    'dunking basketball',
    'dribbling basketball',
    'kicking soccer ball',
    'kicking field goal',
    'passing soccer ball',
]
label_id = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
]
label_to_id = dict(zip(labels, label_id))

problem_title = 'Kinetics video classification'
_target_column_name = 'class'
_prediction_label_names = labels
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.VideoClassifier(
    label_to_id=label_to_id,
)

score_types = [
    rw.score_types.Accuracy(name='acc', precision=3),
    rw.score_types.NegativeLogLikelihood(name='nll', precision=3),
]


def get_cv(folder_X, y):
    _, X = folder_X
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    X = df['id'].values
    y = df['class'].values
    folder = os.path.join(path, 'data', 'frames')
    return (folder, X), y


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)
