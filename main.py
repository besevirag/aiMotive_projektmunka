import os
import cv2
import numpy as np
from src.logger import setup_logger
from src.data_loader import Dataset
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Start process")

    BASE_FOLDER = r"C:\aimotive projektmunka\train\highway"
    CSV_PATH = "./data/id_data.csv"

    #laz np.étrehozunk egy mappát a kimenetnek
    OUTPUT_DIR = "output_samples"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        dataset = Dataset(csv_path=CSV_PATH, folder=BASE_FOLDER, logger=logger)
    except Exception as e:
        logger.error(f"Error: {e}")
        exit()

    processed_count=0
    max_to_process=5

    for i in range(len(dataset)):
        if processed_count>=max_to_process:
            break

        data=dataset[i]
        if data is None:
            continue

        processed_count+=1

        #sdatok kinyerése
        image = data['image']  #[H,W,3]
        depth = data['depth']  #[H,W,1]
        mask = data['gt_mask']  #[H,W,1]

        f_id=dataset.ids.iloc[i, 2]  #frame azonosító a fájlnévhez

        np.save(os.path.join(OUTPUT_DIR, f"frame_{f_id}_depth.npy"), depth)
        np.save(os.path.join(OUTPUT_DIR, f"frame_{f_id}_mask.npy"), mask)

        #Eredeti kép mentése
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{f_id}_img.jpg"), image)

        #Maszk mentése (0-1 tartományt átrakjuk 0-255-re, hogy látszódjon)
        mask_visual=(mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{f_id}_mask.png"), mask_visual)
        """ 
        #Mélységtérkép színezett mentése
        #Csak azokat a pontokat színezzük, ahol van adat
        depth_visual=np.zeros_like(image)
        if np.any(mask>0):
            #Normalizáljuk a mélységet 0-255 közé a látvány kedvéért
            d_min, d_max = depth[mask > 0].min(), depth[mask > 0].max()
            depth_norm = 255 * (depth - d_min) / (d_max - d_min + 1e-8)
            depth_norm = depth_norm.astype(np.uint8)

            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            #Csak ott tartjuk meg a színt, ahol a maszk 1-es
            depth_visual = cv2.bitwise_and(depth_color, depth_color, mask=mask_visual)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{f_id}_depth_color.png"), depth_visual)
        """

        logger.info(f"[{processed_count}/{max_to_process}] Saved: frame_{f_id}")

    logger.info(f"Done")