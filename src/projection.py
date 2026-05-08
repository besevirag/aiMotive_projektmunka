import cv2
import logging
import numpy as np


class CameraProjection:
    def __init__(self, image, extrinsic, intrinsic, lidar_points, logger):
        self.image = image
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.lidar_points = lidar_points
        self.logger = logger

    def lidar_to_camera(self):
        lp = self.lidar_points[:, :3]
        #homogén koordináták (N, 4) a 4x4-es mátrixszorzáshoz
        lp_hom = np.hstack([lp, np.ones((lp.shape[0], 1))])

        #test -> kamera transzformáció
        points_cam = self.extrinsic @ lp_hom.T

        #csak a kamera előtt lévő pontok (Z > 0) megtartása
        valid = points_cam[2, :] > 0
        points_cam_valid = points_cam[:, valid]

        #vetítés a 2D képsíkra
        uvw = self.intrinsic @ points_cam_valid
        #perspektivikus osztás (u, v koordináták kinyerése)
        uv = (uvw[:2] / (uvw[2] + 1e-8)).T
        depths = points_cam_valid[2, :]

        return uv, depths

    def get_projection_matrix(self) -> np.ndarray:
        """
        a tanításhoz/feldolgozáshoz szükséges mátrix
        oszlopok: 0: u (pixel x), 1: v (pixel y), 2: depth (target távolság)
        """
        uv, depths = self.lidar_to_camera()
        h, w = self.image.shape[:2]

        #kiszűrjük azokat a pontokat, amik a képkereten kívülre esnének
        mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & \
               (uv[:, 1] >= 0) & (uv[:, 1] < h)

        uv_valid = uv[mask]
        depths_valid = depths[mask].reshape(-1, 1)

        #összefűzés (N, 3) méretű tömbbé
        return np.hstack([uv_valid, depths_valid])

    def show_points_on_img(self, window_name="Projection"):
        """vetített pontok"""
        uv, depths = self.lidar_to_camera()
        h, w = self.image.shape[:2]

        #mélység normalizálása a színezéshez
        d_min, d_max = depths.min(), depths.max()
        depths_norm = (depths - d_min) / (d_max - d_min + 1e-8)

        for i, (u, v) in enumerate(uv):
            u, v = int(u), int(v)
            if 0 <= u < w and 0 <= v < h:
                d = depths_norm[i]
                color = (0, 255, 0) if d < 0.5 else (0, int(255 * (1 - d)), int(255 * d))
                cv2.circle(self.image, (u, v), 2, color, -1)

        cv2.imshow(window_name, self.image)
        cv2.waitKey(1)