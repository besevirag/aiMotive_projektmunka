import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.logger import setup_logger
from src.data_loader import Dataset

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Start")

    BASE_FOLDER = r"C:\aimotive projektmunka\train\highway"
    CSV_PATH = "./data/id_data.csv"
    OUTPUT_DIR = "batch_test"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #Dataset és DataLoader (batch_size=5)
    dataset = Dataset(csv_path=CSV_PATH, folder=BASE_FOLDER, logger=logger)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    #csak az első batch lekérése
    batch_data = next(iter(train_loader))

    images = batch_data['image']  #[5, 704, 1024, 3]
    depths = batch_data['depth']  #[5, 704, 1024, 1]
    masks = batch_data['gt_mask']  #[5, 704, 1024, 1]

    #végigmegyünk mind az 5 elemen a batch-en belül
    for i in range(5):
        #kinyerjük a konkrét mátrixokat (Tensor->Numpy)
        img_np = images[i].numpy().astype(np.uint8)
        depth_np = depths[i].numpy()
        mask_np = masks[i].numpy()

        prefix = f"sample_{i}"

        np.save(os.path.join(OUTPUT_DIR, f"{prefix}_depth_raw.npy"), depth_np)
        np.save(os.path.join(OUTPUT_DIR, f" {prefix}_mask_raw.npy"), mask_np)

        #eredeti kép
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_image.jpg"), img_np)

        #mask
        mask_visual = (mask_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_mask.png"), mask_visual)
    """
        #színes mélység
        depth_visual = np.zeros_like(img_np)
        if np.any(mask_np > 0):
            d_min, d_max = depth_np[mask_np > 0].min(), depth_np[mask_np > 0].max()
            depth_norm = 255 * (depth_np - d_min) / (d_max - d_min + 1e-8)
            depth_norm = depth_norm.astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_visual = cv2.bitwise_and(depth_color, depth_color, mask=mask_visual)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_depth_view.png"), depth_visual)
    """

    logger.info(f"Sample {i} saved")

    logger.info(f"Done")