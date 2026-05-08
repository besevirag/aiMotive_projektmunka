import os
import laspy
import logging
import json
import numpy as np

from .models import LidarData

EGO_FILTER_X = (-1.2, 3.8)  # (hátsó, első) tengely
EGO_FILTER_Y = (-1.7, 1.7)  # (jobb, bal) tengely


class LidarDataLoader:
    """
    Class a LiDAR pontfelhő adatok betöltéséért és előfeldolgozásáért

    Attributes:
        data (LidarData): A betöltött és feldolgozott LiDAR adatok
    """

    def __init__(
        self,
        folder: str,
        section_id: str,
        frame_id: str,
        logger: logging.Logger
    ):
        """
        Létrehozza a LidarDataLoader objektumot és betölti az adatokat

        Parameters:
            folder: Az adatokat tartalmazó mappa elérési útja
            section_id: A section azonosítója.
            frame_id: A frame azonosítója.
            logger: Python logger nyomonkövetéshez
        """
        self.folder = folder
        self.section_id = section_id
        self.frame_id = frame_id
        self.logger = logger
        self.data = self.load_lidar()

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def load_lidar(self) -> LidarData:
        """
        Betölti a LIDAR pontfelhőt és visszaadja az feldolgozott adatokat

        Returns:
            LidarData: A betöltött, szűrt és egomozgás-kompenzált pontfelhő.
        """
        return self._load_lidar_internal()

    def filter_ego_car(self, pc):
        """
        Kiszűri az járműhöz tartozó LIDAR pontokat

        Parameters:
            pc: Bemeneti pontfelhő ((N, 5) tömb: X, Y, Z, intenzitás, GPS idő)

        Returns:
            np.ndarray: A szűrt pontfelhő, egójármű pontok nélkül
        """
        filter_x = (pc[:, 0] > EGO_FILTER_X[0]) & (pc[:, 0] < EGO_FILTER_X[1])
        filter_y = (pc[:, 1] > EGO_FILTER_Y[0]) & (pc[:, 1] < EGO_FILTER_Y[1])
        valid = ~(filter_x & filter_y)

        return pc[valid]

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _load_lidar_internal(self) -> LidarData:
        """
        Beolvassa a LAZ fájlt és elvégzi a szűrést

        Returns:
            LidarData: Az feldolgozott pontfelhőt tartalmazó adatok
        """
        path = self._get_lidar_path()
        self.logger.info(f"Loading LIDAR files from: '{path}'")

        try:
            las = laspy.read(path)
            lidar_pcd = np.array(
                [las.x, las.y, las.z, las.intensity, las.gps_time],
                dtype=np.float32)
            lidar_pcd = lidar_pcd.T
        except Exception as e:
            self.logger.error(f"Error reading LIDAR data: {path} - {e}")
            raise

        self.logger.debug(f"Size of pointcloud: {lidar_pcd.shape[0]} points")

        egomotion_path = self._get_egomotion_path()
        self.logger.info(f"Loading egomotion JSON from: '{egomotion_path}'")

        with open(egomotion_path) as f:
            egomotion = json.load(f)

        RT_main_frame = np.array(
            egomotion[str(int(self.frame_id))]
        ).reshape(4, 4)

        RT_current = np.array(egomotion[str(int(self.frame_id))]).reshape(4, 4)
        RT_transform = np.linalg.inv(RT_main_frame) @ RT_current
        self.logger.debug("Egomotion transformation matrix calculated")

        lidar_data = self.filter_ego_car(lidar_pcd)
        lidar_data_coords = np.hstack(
            [lidar_data[:, :3],
             np.ones((lidar_data.shape[0], 1))]
        )
        lidar_data[:, :3] = (lidar_data_coords @ RT_transform.T)[:, :3]

        self.logger.info(
            f"LIDAR data loaded: {lidar_data.shape[0]}"
        )

        return LidarData(
            idx=(self.section_id, self.frame_id),
            data=lidar_data
        )

    def _get_lidar_path(self) -> str:
        """
        Összeállítja a LAZ pontfelhő fájl teljes elérési útját

        Returns:
            str: A LAZ fájl teljes elérési útja
        """
        return os.path.join(
            self.folder,
            self.section_id,
            "dynamic",
            "raw-revolutions",
            f"frame_{self.frame_id}.laz"
        )

    def _get_egomotion_path(self) -> str:
        """
        Összeállítja az egomotion JSON fájl teljes elérési útját

        Returns:
            str: Az egomotion JSON fájl teljes elérési útja
        """
        return os.path.join(
            self.folder,
            self.section_id,
            "sensor",
            "gnssins",
            "egomotion.json"
        )
