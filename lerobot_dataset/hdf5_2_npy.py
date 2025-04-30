#!/usr/bin/env python
import glob

import cv2
import h5py
import numpy as np
import re


def convert(idx, path):
    with h5py.File(path, 'r') as f:
        qpos = f['observations']['qpos'][:]

        episode = []
        for step in range(len(qpos)):
            img = cv2.imdecode(np.frombuffer(f['observations']['images']['cam_right_wrist'][step], np.uint8), cv2.IMREAD_COLOR)
            episode.append({
                'image': img,
                'state': np.asarray(qpos[step], dtype=np.float32),
                'action': np.asarray(f['action'][step], dtype=np.float32),
                'language_instruction': 'Put the yellow toy block in a stainless steel bowl.',
            })
        np.save("/home/ubuntu/Downloads/openvla/npy_right_left/episode_%s.npy" % idx, episode)
        print(path)


if __name__ == "__main__":
    episode_paths = glob.glob("/home/ubuntu/Downloads/lerobot/hdf5_right_left/episode_*.hdf5")

    # for smallish datasets, use single-thread parsing
    for sample in episode_paths:
        numbers = re.findall(r'episode_(\d+).hdf5', sample)
        convert(numbers[0], sample)

