from dataclasses import dataclass

import numpy as np


@dataclass
class CameraParams:
    """
    Kamera paramétereihez tartozó class

    Attributes:
        model: kamera model (pinhole, fisheye)
        intrinsic: intrinsic mátrix
        extrinsic: extrinsic mátrix
        dist: distortion koefficiensek
    """

    model: str
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    dist: np.array


@dataclass
class CameraData:
    """
    Kép adatokhoz tartozó class

    Attributes:
        idx: index a kép adatok betöltésére Tuple(section_id, frame_id)
        camera_id: melyik kamera képéhez tartozó adatokat látjuk (front, back, left, right)
        data: kép adatok NumPy tömbként
    """

    idx: tuple[str, str]
    camera_id: str
    data: np.ndarray
    params: CameraParams


@dataclass
class LidarData:
    """
    LIDAR adatokhoz tartozó class

    Attributes:
        idx: index a LIDAR adatok betöltésére Tuple(section_id, frame_id)
        data: LIDAR adatok NumPy tömbként
    """

    idx: tuple[str, str]
    data: np.ndarray
