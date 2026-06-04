import os
import torch
import logging

import segmentation_models_pytorch as smp
import pandas as pd

from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler
from src.logger import setup_logger
from src.data_loader import Dataset
from train import train_epoch

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

BASE_FOLDER = "/mnt/oldssd/aimotive-dataset/train/"
CSV_PATH = "./data/id_data.csv"
OUTPUT_DIR = "batch_test"
CHECKPOINT_DIR = "weights"
DATA_DIRS = ["highway", "night", "rain", "urban"]

EPOCHS = 5
BATCH_SIZE = 7
GRAD_ACCUM_STEPS = 1
LR = 1e-3


def save_checkpoint(
        model,
        optimizer,
        scaler,
        epoch: int,
        loss: float,
        logger
) -> None:
    """Modell állapot mentése .pt fájlba."""
    path = os.path.join(
        CHECKPOINT_DIR,
        f"epoch_{epoch:03d}_loss_{loss:.4f}.pt"
    )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "loss":                 loss,
    }, path)
    logger.info(f"Checkpoint mentve: {path}")


if __name__ == "__main__":
    logger = setup_logger(level=logging.INFO)
    logger.info("Start")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    logger.info("Model loaded")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda")

    datasets = [
        Dataset(
            csv_path=CSV_PATH,
            folder=os.path.join(BASE_FOLDER, d),
            logger=logger
        ) for d in ["highway", "night", "rain", "urban"]
    ]

    combined_dataset = ConcatDataset(datasets)

    train_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    total_losses_per_epoch = pd.DataFrame()

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"==================== Epoch: {epoch}/{EPOCHS} ====================")

        logger.info(f"Dataset mérete: {len(combined_dataset)}")
        logger.info(f"Batch-ek száma: {len(train_loader)}")

        losses_per_epoch, train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            logger=logger
        )
        logger.info(f"Train loss: {train_loss:.4f} m")

        scheduler.step()
        logger.info(f"LR: {scheduler.get_last_lr()[0]:.2e}")
        total_losses_per_epoch[f"EPOCH_{epoch}"] = losses_per_epoch

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            loss=train_loss,
            logger=logger
        )

        total_losses_per_epoch.to_csv(
            f"losses_{epoch}.csv",
            index=False
        )

    logger.info("Done")
