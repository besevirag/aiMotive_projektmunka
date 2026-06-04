import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from src.logger import setup_logger
from src.data_loader import Dataset

if __name__=="__main__":
    logger=setup_logger()
    logger.info("Error heatmap generálása")

    BASE_FOLDER=r"C:\aimotive projektmunka\train\highway"
    CSV_PATH="./data/id_data.csv"
    OUTPUT_DIR="heatmap_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)

    checkpoint_path="weights/epoch_005_loss_4.5506.pt"

    if not os.path.exists(checkpoint_path):
        logger.error(f"Nem található a súlyfájl a megadott helyen: {checkpoint_path}")
        exit()

    checkpoint=torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset=Dataset(csv_path=CSV_PATH, folder=BASE_FOLDER, logger=logger)

    #teszteljünk le pl az első 3 képet
    for sample_idx in range(3):
        logger.info(f"Feldolgozás: {sample_idx}")
        data=dataset[sample_idx]

        frame_id=dataset.valid_df.iloc[sample_idx]["frame_id"]

        #adatok előkészítése a hálónak (dimenzió: [1, C, H, W])
        image_tensor=data['image'].unsqueeze(0).to(device)
        target_depth=data['depth'].to(device)
        mask=data['gt_mask'].to(device)

        with torch.no_grad():
            pred_depth=model(image_tensor).squeeze(0)

        #visszaalakítjuk numpy tömbbé a képi feldolgozáshoz
        pred_np=pred_depth.cpu().numpy().squeeze()
        target_np=target_depth.cpu().numpy().squeeze()
        mask_np=mask.cpu().numpy().squeeze()

        orig_img=data['image'].cpu().numpy().transpose(1, 2, 0)
        orig_img=(orig_img * 255).astype(np.uint8)

        error_map=np.abs(pred_np-target_np)

        error_map[mask_np==0]=0.0

        #összehasonlíthatóság kedvéért a max hibahatárt 15 méterre állítjuk pl., e fölött fixen piros lesz
        MAX_ERROR=15.0
        error_clipped=np.clip(error_map, 0, MAX_ERROR)
        error_norm=(error_clipped/MAX_ERROR*255).astype(np.uint8)

        #kék=nincs hiba, piros=nagy hiba (jet skála)
        error_heatmap=cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)

        #háttér, ahol nincs lidar:
        error_heatmap[mask_np==0]=0

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id}_01_org.jpg"), orig_img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id}_02_error_heatmap.png"), error_heatmap)

        logger.info(f"Done.")

    logger.info(f"Minden hőtérkép mentve.")