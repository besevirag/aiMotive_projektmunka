import pandas as pd
import numpy as np
import os
import logging
from .camera_loader import CameraDataLoader
from .lidar_loader import LidarDataLoader
from .projection import CameraProjection


class Dataset:
    def __init__(self, csv_path: str, folder: str, logger):
        self.folder = folder
        self.logger = logger

        df = pd.read_csv(csv_path, dtype=str)

        #csak azokat a sorokat tartjuk meg, ahol a mappa létezik
        self.valid_ids = []
        self.logger.info("Filtering csv for existing files...")

        for i in range(len(df)):
            section_id = df.iloc[i, 1]
            if os.path.exists(os.path.join(self.folder, section_id)):
                self.valid_ids.append(df.iloc[i])

        self.logger.info(f"Filtering: {len(self.valid_ids)} valid samples.")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        row = self.valid_ids[idx]
        section_id = row.iloc[1]
        frame_id = row.iloc[2]

        #szenzorok betöltése
        lidar_loader = LidarDataLoader(self.folder, section_id, frame_id, self.logger)
        camera_loader = CameraDataLoader(self.folder, section_id, frame_id, self.logger)

        cam = camera_loader.front_camera
        img = cam.data  #[H,W,3]
        h, w = img.shape[:2]

        projection = CameraProjection(
            image=img,
            extrinsic=cam.params.extrinsic,
            intrinsic=cam.params.intrinsic,
            lidar_points=lidar_loader.data.data,
            logger=self.logger
        )

        proj_matrix = projection.get_projection_matrix()
        depth_gt, depth_mask = self.rasterize_to_image(proj_matrix, h, w)

        #(H, W, C)
        return {
            'image': img.astype(np.float32),
            'depth': depth_gt,
            'gt_mask': depth_mask
        }

    def rasterize_to_image(self, proj_matrix, h, w):
        """
        A 2D-re vetített pontokból sűrűbb (rasterized) mátrixot készít.
        """
        depth_map = np.zeros((h, w, 1), dtype=np.float32)
        mask = np.zeros((h, w, 1), dtype=np.float32)

        if proj_matrix is not None and len(proj_matrix) > 0:
            #koordináták kerekítése egészekre a pixelekhez
            u = np.round(proj_matrix[:, 0]).astype(int)
            v = np.round(proj_matrix[:, 1]).astype(int)
            depths = proj_matrix[:, 2]

            #csak a képkereten belüli pontokat tartjuk meg
            valid_indices = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u, v, depths = u[valid_indices], v[valid_indices], depths[valid_indices]

            #értékek beírása a mátrixokba
            depth_map[v, u, 0] = depths
            mask[v, u, 0] = 1.0

        return depth_map, mask