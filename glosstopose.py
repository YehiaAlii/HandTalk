import os
import re
from typing import List, Tuple
import numpy as np
import scipy.signal
from scipy.spatial.distance import cdist
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, correct_wrists, pose_normalization_info
from pose_format.numpy import NumPyPoseBody
from num2words import num2words


def normalize_pose(pose: Pose) -> Pose:
    return pose.normalize(pose_normalization_info(pose.header))


def trim_pose(pose: Pose, start=True, end=True):
    if len(pose.body.data) == 0:
        return pose

    wrist_indexes = [
        pose.header._get_point_index('LEFT_HAND_LANDMARKS', 'WRIST'),
        pose.header._get_point_index('RIGHT_HAND_LANDMARKS', 'WRIST')
    ]
    either_hand = pose.body.confidence[:, 0, wrist_indexes].sum(axis=1) > 0

    first_non_zero_index = np.argmax(either_hand) if start else 0
    last_non_zero_index = (
        len(either_hand) - np.argmax(either_hand[::-1]) - 1) if end else len(either_hand)

    pose.body.data = pose.body.data[first_non_zero_index:last_non_zero_index]
    pose.body.confidence = pose.body.confidence[first_non_zero_index:last_non_zero_index]
    return pose


def concatenate_poses(poses: List[Pose]) -> Pose:
    # print('Reducing poses...')
    poses = [reduce_holistic(p) for p in poses]

    # print('Normalizing poses...')
    poses = [normalize_pose(p) for p in poses]

    # Trim the poses to only include the parts where the hands are visible
    # print('Trimming poses...')
    poses = [trim_pose(p, i > 0, i < len(poses) - 1)
             for i, p in enumerate(poses)]

    # Concatenate all poses
    # print('Smooth concatenating poses...')
    pose = smooth_concatenate_poses(poses)

    # print('Correcting wrists...')
    pose = correct_wrists(pose)

    # Scale the newly created pose
    # print('Scaling pose...')
    new_width = 512
    shift = 1.25
    shift_vec = np.full(
        shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    pose.body.data = (pose.body.data + shift_vec) * new_width
    pose.header.dimensions.height = pose.header.dimensions.width = int(
        new_width * shift * 2)

    return pose


# lookup
class PoseLookup:
    def __init__(self, directory: str, language: str):
        with open(os.path.join(directory, 'words.txt'), mode='r', encoding='utf-8') as f:
            words = f.readlines()

        self.glosses = set(word.replace("\n", "") for word in words)
        self.directory = directory
        self.language = language

    def read_pose(self, pose_path: str):
        pose_path = os.path.join(
            self.directory, self.language, pose_path + ".pose")
        with open(pose_path, "rb") as f:
            return Pose.read(f.read())

    def lookup(self, word: str) -> Pose:
        word = word.lower().strip()
        if word in self.glosses:
            return self.read_pose(word)

    def lookup_sequence(self, glosses: List[str]) -> Tuple[List[Pose], List[str]]:
        poses: List[Pose] = []
        words: List[str] = []

        for gloss in glosses:
            pose = self.lookup(gloss)
            if pose:
                poses.append(pose)
                words.append(gloss)
            else:
                for char in gloss:
                    pose = self.lookup(char)
                    if pose:
                        poses.append(pose)
                        words.append(char)

        return poses, words

    def gloss_to_pose(self, glosses: List[str]) -> Tuple[Pose, List[str]]:
        # Transform the list of glosses into a list of poses
        poses, words = self.lookup_sequence(glosses)

        if poses:
            # Concatenate the poses to create a single pose
            return concatenate_poses(poses), words

        return None, None


# smoothing
def pose_savgol_filter(pose: Pose):
    [face_component] = [c for c in pose.header.components if c.name == 'FACE_LANDMARKS']
    face_range = range(
        pose.header._get_point_index(
            'FACE_LANDMARKS', face_component.points[0]),
        pose.header._get_point_index(
            'FACE_LANDMARKS', face_component.points[-1]),
    )

    _, _, points, dims = pose.body.data.shape
    for p in range(points):
        if p not in face_range:
            for d in range(dims):
                pose.body.data[:, 0, p, d] = scipy.signal.savgol_filter(
                    pose.body.data[:, 0, p, d], 3, 1)
    return pose


def create_padding(time: float, example: Pose) -> NumPyPoseBody:
    fps = example.body.fps
    padding_frames = int(time * fps)
    data_shape = example.body.data.shape
    return NumPyPoseBody(fps=fps,
                         data=np.zeros(
                             shape=(padding_frames, data_shape[1], data_shape[2], data_shape[3])),
                         confidence=np.zeros(shape=(padding_frames, data_shape[1], data_shape[2])))


def s_concatenate_poses(poses: List[Pose], padding: NumPyPoseBody, interpolation='linear') -> Pose:
    # Add padding to all poses except the last one
    for pose in poses[:-1]:
        pose.body.data = np.concatenate((pose.body.data, padding.data))
        pose.body.confidence = np.concatenate(
            (pose.body.confidence, padding.confidence))

    # Concatenate all tensors
    new_data = np.concatenate([pose.body.data for pose in poses])
    new_conf = np.concatenate([pose.body.confidence for pose in poses])
    new_body = NumPyPoseBody(
        fps=poses[0].body.fps, data=new_data, confidence=new_conf)
    new_body = new_body.interpolate(kind=interpolation)
    return Pose(header=poses[0].header, body=new_body)


def find_best_connection_point(pose1: Pose, pose2: Pose, window=0.3):
    p1_size = int(len(pose1.body.data) * window)
    p2_size = int(len(pose2.body.data) * window)

    last_data = pose1.body.data[len(pose1.body.data) - p1_size:]
    first_data = pose2.body.data[:p2_size]

    last_vectors = last_data.reshape(len(last_data), -1)
    first_vectors = first_data.reshape(len(first_data), -1)

    distances_matrix = cdist(last_vectors, first_vectors, 'euclidean')
    min_index = np.unravel_index(
        np.argmin(distances_matrix, axis=None), distances_matrix.shape)
    last_index = len(pose1.body.data) - p1_size + min_index[0]
    return last_index, min_index[1]


def smooth_concatenate_poses(poses: List[Pose], padding=0.20) -> Pose:
    if len(poses) == 1:
        return poses[0]

    start = 0
    for i, pose in enumerate(poses):
        # print('Processing', i + 1, 'of', len(poses), '...')
        if i != len(poses) - 1:
            end, next_start = find_best_connection_point(
                poses[i], poses[i + 1])
        else:
            end = len(pose.body.data)
            next_start = None

        pose.body = pose.body[start:end]
        start = next_start

    padding_pose = create_padding(padding, poses[0])
    # print('Concatenating...')
    single_pose = s_concatenate_poses(poses, padding_pose)
    # print('Smoothing...')
    return pose_savgol_filter(single_pose)


# utils
def scale_down(pose: Pose, value: int = 256):
    scale = pose.header.dimensions.width / value
    pose.header.dimensions.width = int(pose.header.dimensions.width / scale)
    pose.header.dimensions.height = int(pose.header.dimensions.height / scale)
    pose.body.data = pose.body.data / scale


def scale_up(pose: Pose, value: int = 2):
    pose.body.data *= value
    pose.header.dimensions.width *= value
    pose.header.dimensions.height *= value


def prepare_glosses(sentence: str) -> List[str]:
    glosses: List[str] = re.findall(r'\b[a-zA-Z0-9]+\b', sentence.lower())

    for i, word in enumerate(glosses):
        if word.isdigit():
            number_words = num2words(int(word)).split()
            glosses[i:i+1] = number_words

    return glosses