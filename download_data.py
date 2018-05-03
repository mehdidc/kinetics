from joblib import Parallel, delayed
from itertools import chain
from subprocess import call
import os
from utils.download import main as download_videos
from utils.download import construct_video_filename
from problem import labels, train_size, test_size
from glob import glob


def _remove_non_existent(ds):
    ds['exists'] = [
        len(glob(os.path.join('data', 'frames', ds.iloc[i]['id'], '*.jpg'))) > 0
        for i in range(len(ds))
    ]
    return ds[ds['exists']]


def _video_id(row, folder):
    filename = construct_video_filename(row, folder)
    filename = filename.replace(folder, '')
    filename = filename.replace(os.sep, '')
    return filename


def _build_frames(filename, dest):
    cmd = 'ffmpeg -i {} {}/image_%010d.jpg'.format(filename, dest)
    call(cmd, shell=True)


if __name__ == '__main__':
    # Download CSVs
    url_prefix = 'https://raw.githubusercontent.com/activitynet/ActivityNet/master/Crawler/Kinetics/data'
    if not os.path.exists('data/full_train.csv'):
        call('wget {}/kinetics-600_train.csv --output-document=data/full_train.csv'.format(url_prefix), shell=True)
    if not os.path.exists('data/full_test.csv'):
        call('wget {}/kinetics-600_val.csv --output-document=data/full_test.csv'.format(url_prefix), shell=True)
    
    # Download videos and build train.csv and test.csv
    
    # Train data
    train_ds = download_videos(
        'data/full_train.csv', 
        'data/videos', 
        nb_examples=train_size, 
        labels=labels,
        num_jobs=1,
    )

    train_ds['id'] = [
        _video_id(train_ds.iloc[i], 'data/videos') 
        for i in range(len(train_ds))
    ]
    train_ds['class'] = train_ds['label-name']
    train_ds = train_ds[['id', 'class']]
    # Test data
    test_ds = download_videos(
        'data/full_test.csv', 
        'data/videos', 
        nb_examples=test_size, 
        labels=labels,
        num_jobs=1,
    )
    test_ds['id'] = [
        _video_id(test_ds.iloc[i], 'data/videos')
        for i in range(len(test_ds))
    ]
    test_ds['class'] = test_ds['label-name']
    test_ds = test_ds[['id', 'class']]
    
    # Build frames

    args = []
    for id_ in chain(train_ds['id'].values, test_ds['id'].values):
        filename = os.path.join('data', 'videos', id_)
        dest = filename.replace('videos', 'frames')
        if os.path.exists(dest):
            continue
        os.makedirs(dest)
        args.append((filename, dest))
    Parallel(n_jobs=-1)(delayed(_build_frames)(*a) for a in args)

    # write csv
    train_ds = _remove_non_existent(train_ds)
    test_ds = _remove_non_existent(test_ds)
    train_ds.to_csv('data/train.csv', index=False)
    test_ds.to_csv('data/test.csv', index=False)
