import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.logger import setup_logger
from src.data_loader import Dataset

def lossfunc(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    kiszámoljuk a maszkolt l1 veszteséget a ritka mélységtérképen.
    preds: háló tippje [Batch, 1, H, W]
    targets: valódi mélység (depth) [Batch, 1, H, W]
    masks: bináris maszk (gt_mask) [Batch, 1, H, W]
    """
    absolute_error=torch.abs(preds-targets)
    masked_error=absolute_error*masks

    total_loss=torch.sum(masked_error)
    valid_pixels=torch.sum(masks)

    if valid_pixels==0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    return total_loss/valid_pixels

def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 5
):
    # model.train()
    running_loss = .0

    for i, data in enumerate(train_loader):
        images = data["image"].to(device)
        depths = data["depth"].to(device)
        masks = data["gt_mask"].to(device)

        optimizer.zero_grad()
        # preds = model.predict(images)
        # optimizer.step()

        # running_loss += loss.item()


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Start")

    BASE_FOLDER = r"C:\aimotive projektmunka\train\highway"
    CSV_PATH = "./data/id_data.csv"
    OUTPUT_DIR = "batch_test"
    EPOCHS = 5
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = model().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Dataset és DataLoader (batch_size=5)
    dataset = Dataset(csv_path=CSV_PATH, folder=BASE_FOLDER, logger=logger)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    # csak az első batch lekérése
    batch_data = next(iter(train_loader))

    images = batch_data['image']   # [5, 3, 704, 1024]
    depths = batch_data['depth']   # [5, 3, 704, 1024]
    masks = batch_data['gt_mask']  # [5, 3, 704, 1024]

    logger.info("random frame:")

    random_predictions = torch.rand_like(depths)*50.0

    #mask l1 loss
    loss = lossfunc(random_predictions, depths, masks)

    images = images.to(device)
    depths = depths.to(device)
    masks = masks.to(device)


    for i in range(5):
        img = images[i]
        depth = depths[i]
        mask = masks[i]

        logger.debug(f"Image data: {img}")
        logger.debug(f"Image shape: {img.shape}")

        logger.debug(f"Depth data: {depth}")
        logger.debug(f"Depth shape: {depth.shape}")

        logger.debug(f"Mask data: {mask}")
        logger.debug(f"Mask shape: {mask.shape}")

        prefix = f"sample_{i}"

        # np.save(os.path.join(OUTPUT_DIR, f"{prefix}_depth_raw.npy"), depth)
        # np.save(os.path.join(OUTPUT_DIR, f" {prefix}_mask_raw.npy"), mask)

        # eredeti kép
        # cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_image.jpg"), img)

        # mask
        # mask_visual = (mask_np * 255).astype(np.uint8)
        # cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_mask.png"), mask_visual)
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
    logger.info(f"Loss: {loss.item():.4f} méter")
    logger.info("Done")
