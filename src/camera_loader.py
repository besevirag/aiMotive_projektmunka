import os
import cv2
import logging
import json
import numpy as np
from typing import Any
from .models import CameraData, CameraParams

CAMERA_TABLE = {
    "back_camera": "B_MIDRANGECAM_C",
    "front_camera": "F_MIDLONGRANGECAM_CL",
    "left_camera": "M_FISHEYE_L",
    "right_camera": "M_FISHEYE_R"
}


class CameraDataLoader:
    def __init__(self, folder: str, section_id: str, frame_id: str, logger: logging.Logger):
        self.folder = folder
        self.section_id = section_id
        self.frame_id = frame_id
        self.logger = logger

        self.front_camera = CameraData(
            idx=(section_id, frame_id),
            camera_id="front_camera",
            data=self.load_camera("front_camera"),
            params=self.read_camera_params("front_camera")
        )

        self.back_camera = None
        self.left_camera = None
        self.right_camera = None

        """
        #ha később kellenének:
        self.back_camera = CameraData(
            idx=(section_id, frame_id),
            camera_id="back_camera",
            data=self.load_camera("back_camera"),
            params=self.read_camera_params("back_camera")
        )
        self.left_camera = CameraData(
            idx=(section_id, frame_id),
            camera_id="left_camera",
            data=self.load_camera("left_camera"),
            params=self.read_camera_params("left_camera")
        )
        self.right_camera = CameraData(
            idx=(section_id, frame_id),
            camera_id="right_camera",
            data=self.load_camera("right_camera"),
            params=self.read_camera_params("right_camera")
        )
        """

    def load_camera(self, camera_id: str) -> np.ndarray:
        return self._load_camera_internal(camera_id)

    def read_camera_params(self, camera: str) -> CameraParams:
        params = self._load_calibration_data()[CAMERA_TABLE[camera]]
        model = params["model"]
        intrinsic = self._get_intrinsic(params["focal_length_px"], params["principal_point_px"])
        extrinsic = params["RT_sensor_from_body"]
        dist = np.array(params.get("distortion_coeffs", [0, 0, 0, 0]), dtype=np.float32)

        return CameraParams(model=model, intrinsic=intrinsic, extrinsic=extrinsic, dist=dist)

    def _load_camera_internal(self, camera_id: str) -> np.ndarray: #létezik-e a fájl
        camera = CAMERA_TABLE[camera_id]
        path = self._get_camera_path(camera)

        if not os.path.exists(path):
            self.logger.error(f"Camera file not found: '{path}'")
            raise FileNotFoundError(f"File not found: {path}")

        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            self.logger.error(f"Failed to read image data: {path} - {e}")
            raise

        if image is None:
            raise ValueError(f"Could not load image: {path}")

        return image

    def _get_camera_path(self, camera: str) -> str:
        return os.path.join(self.folder, self.section_id, "sensor", "camera", camera, f"{camera}_{self.frame_id}.jpg")

    def _get_intrinsic(self, f: list[float], p: list[float]) -> np.ndarray:
        return np.array([[f[0], 0, p[0], 0], [0, f[1], p[1], 0], [0, 0, 1, 0]])

    def _load_calibration_data(self) -> dict[str, Any]:
        path = os.path.join(self.folder, self.section_id, "sensor", "calibration", "calibration.json")
        with open(path, "r") as f:
            return json.load(f)