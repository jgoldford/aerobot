#!/usr/bin/env python
from aerobot.io import download_training_data, ASSET_PATH

import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download training data')
    parser.add_argument('-f', '--force', action='store_true', help='blow away existing files.')
    args = parser.parse_args()

    if args.force:
        print('removing existing files...')
        fnames = ['train/training_data.tar.gz', 'train/training_data.h5']
        fpaths = [os.path.join(ASSET_PATH, x) for x in fnames]
        for fpath in fpaths:
            if os.path.exists(fpath):
                print('removing {}'.format(fpath))
                os.remove(fpath)

    download_training_data()