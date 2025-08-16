import os
import time
import yaml
import shutil
import torch
import wandb
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import ShapeNetDataset
from models.pointnet2 import PointNet2SemSeg
from utils.metrics import compute_mIoU, compute_precision, compute_recall, compute_f1
from glob import glob
from utils.wandb_tools import log_pointcloud_to_wandb

# Dataset split management

def check_and_split_dataset(config):
    from glob import glob

    points_path = Path(config['data']['train_points'])
    labels_path = Path(config['data']['train_labels'])
    split_ratio = config['data']['split']
    splits = ['train', 'val', 'test']

    def check_parallel(folder1, folder2):
        f1 = {Path(x).stem for x in glob(str(folder1 / "*.npy"))}
        f2 = {Path(x).stem for x in glob(str(folder2 / "*.seg"))}
        return f1 == f2

    # Paso 1: revertir si los splits existen pero no están alineados
    if all((points_path / s).exists() and (labels_path / s).exists() for s in splits):
        if all(check_parallel(points_path / s, labels_path / s) for s in splits):
            print("✅ Using existing train/val/test folders.")
            return
        else:
            print("⚠️ Split folders exist but are not aligned. Reverting and re-splitting.")
            for s in splits:
                for f in glob(str(points_path / s / "*.npy")):
                    shutil.move(f, points_path / Path(f).name)
                for f in glob(str(labels_path / s / "*.seg")):
                    shutil.move(f, labels_path / Path(f).name)
                shutil.rmtree(points_path / s)
                shutil.rmtree(labels_path / s)

    print("🔄 Splitting dataset with filename matching...")

    # Paso 2: obtener archivos base
    point_files = {Path(f).stem: Path(f) for f in glob(str(points_path / "*.npy"))}
    label_files = {Path(f).stem: Path(f) for f in glob(str(labels_path / "*.seg"))}

    common_stems = sorted(set(point_files.keys()) & set(label_files.keys()))
    if not common_stems:
        raise RuntimeError(f"❌ No matching point-label pairs found.\n"
                           f"Points: {sorted(point_files.keys())[:5]}\n"
                           f"Labels: {sorted(label_files.keys())[:5]}")

    total = len(common_stems)
    train_end = int(split_ratio['train'] * total)
    val_end = train_end + int(split_ratio['val'] * total)

    split_map = {
        'train': common_stems[:train_end],
        'val': common_stems[train_end:val_end],
        'test': common_stems[val_end:]
    }

    for s in splits:
        (points_path / s).mkdir(parents=True, exist_ok=True)
        (labels_path / s).mkdir(parents=True, exist_ok=True)
        for stem in split_map[s]:
            shutil.move(str(point_files[stem]), str(points_path / s / f"{stem}.npy"))
            shutil.move(str(label_files[stem]), str(labels_path / s / f"{stem}.seg"))

    print("✅ Dataset split complete and aligned by filename.")

### ------------------------ MAIN FUNCTION ------------------------

def main():

    from dataset.dataset import get_min_num_points

    # Load .env
    load_dotenv()
    if not os.path.exists('.env'):
        print("Warning: .env file not found.")
    if 'WANDB_API_KEY' not in os.environ:
        print("Warning: WANDB_API_KEY not found in environment variables.")

    # Load YAML config
    with open("config/train.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Check and split dataset if needed
    check_and_split_dataset(config)

    num_points_ok = True
    min_points = get_min_num_points(config['data']['train_points'] + '/train')
    print(f"✅ Minimum number of points detected: {min_points}")

    # Check if the minimum number of points is sufficient
    if min_points < config['training']['num_points']:
        print(f"❌ Minimum points ({min_points}) is less than required ({config['training']['num_points']}).")
        print(f"✅ Reducing num_points to the minimum found in the dataset {min_points}.")
        num_points_ok = False

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("Warning: Using CPU.")
        device = torch.device("cpu")

    # Dataloaders
    def get_loader(split):
        return DataLoader(
            ShapeNetDataset(
                x_path=os.path.join(config['data']['train_points'], split),
                y_path=os.path.join(config['data']['train_labels'], split),
                num_points=config['training']['num_points'] if num_points_ok else min_points
            ),
            batch_size=config['training']['batch_size'],
            shuffle=(split == "train"),
            num_workers=0,
            drop_last=True
        )

    train_loader = get_loader("train")
    
    val_dataset = ShapeNetDataset(
        x_path=os.path.join(config['data']['train_points'], "val"),
        y_path=os.path.join(config['data']['train_labels'], "val"),
        num_points=config['training']['num_points'] if num_points_ok else min_points
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    from dataset.dataset import detect_num_classes

    train_labels_dir = os.path.join(config['data']['train_labels'], 'train')
    num_classes = detect_num_classes(train_labels_dir)
    print(f"✅ Detected {num_classes} classes in training set.")

    model = PointNet2SemSeg(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # TensorBoard
    writer = SummaryWriter(log_dir=config['training']['log_dir'])

    # wandb
    use_wandb = False
    if config['training'].get('use_wandb', False):
        try:
            import wandb
            if os.getenv('WANDB_API_KEY'):
                os.environ['WANDB_MODE'] = os.getenv('WANDB_MODE', 'online')
                wandb.login(key=os.getenv('WANDB_API_KEY'))
            wandb.init(project=config['experiment_name'], config=config)
            use_wandb = True
        except Exception as e:
            print(f"[WARN] wandb.init() failed: {e}")
            use_wandb = False

    # Training loop
    start_time = time.time()
    global_step = 0

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        all_preds, all_labels = [], []

        for i, (points, labels) in enumerate(train_loader):
            points = points.transpose(1, 2).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            preds = outputs.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            print(f"\rEpoch {epoch+1}/{config['training']['epochs']} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end='', flush=True)

            if use_wandb:
                wandb.log({"Train/Loss_Batch": loss.item()}, step=global_step)

        avg_loss = total_loss / len(train_loader)
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)

        miou = compute_mIoU(all_preds_tensor, all_labels_tensor, num_classes=num_classes)
        precision = compute_precision(all_preds_tensor, all_labels_tensor)
        recall = compute_recall(all_preds_tensor, all_labels_tensor)
        f1 = compute_f1(all_preds_tensor, all_labels_tensor)

        # Validation
        model.eval()
        val_preds, val_labels, val_loss = [], [], 0
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.transpose(1, 2).to(device)
                labels = labels.to(device)
                outputs = model(points)
                val_loss += criterion(outputs, labels).item()
                val_preds.append(outputs.argmax(dim=1).cpu())
                val_labels.append(labels.cpu())

        val_preds = torch.cat(val_preds, dim=0)
        val_labels = torch.cat(val_labels, dim=0)

        val_miou = compute_mIoU(val_preds, val_labels, num_classes=num_classes)
        val_precision = compute_precision(val_preds, val_labels)
        val_recall = compute_recall(val_preds, val_labels)
        val_f1 = compute_f1(val_preds, val_labels)
        val_loss /= len(val_loader)

        elapsed = time.time() - start_time
        epoch_time = time.time() - epoch_start
        remaining = (elapsed / (epoch + 1)) * (config['training']['epochs'] - epoch - 1)

        print(f"\rEpoch {epoch+1}/{config['training']['epochs']} - Loss: {avg_loss:.4f} - mIoU: {miou:.4f} - F1: {f1:.4f} - Val Loss: {val_loss:.4f} - Val mIoU: {val_miou:.4f} - Time: {epoch_time:.1f}s - Remaining: {remaining/60:.1f} min")

        wandb.log({
            "Train/Loss_Epoch": avg_loss,
            "Train/mIoU_Epoch": miou,
            "Train/Precision_Epoch": precision,
            "Train/Recall_Epoch": recall,
            "Train/F1_Epoch": f1
        })

        wandb.log({
            "Val/Loss_Epoch": val_loss,
            "Val/mIoU_Epoch": val_miou,
            "Val/Precision_Epoch": val_precision,
            "Val/Recall_Epoch": val_recall,
            "Val/F1_Epoch": val_f1
        })

        # Extraemos la PRIMERA muestra, no un batch
        val_points, val_labels = val_dataset[0]

        from utils.preprocess import sample_fixed_num_points

        val_points, val_labels = val_dataset[0]
        val_points_np, val_labels_np = sample_fixed_num_points(
            val_points.numpy(), val_labels.numpy(), num_points=config['training']['num_points'] if num_points_ok else min_points
        )

        val_points_batch = torch.tensor(val_points_np).unsqueeze(0).transpose(1, 2).to(device)
        val_preds = model(val_points_batch).argmax(dim=2).cpu().numpy()

        points = val_points_np
        gt_labels = val_labels_np
        pred_labels = val_preds[0]

        print("")
        print(f"[INFO] Ground Truth classes: {np.unique(gt_labels)}")
        print(f"[INFO] Predicted classes:    {np.unique(pred_labels)}")
        print(f"[INFO] Num GT classes:       {len(np.unique(gt_labels))}")
        print(f"[INFO] Num Predicted classes:{len(np.unique(pred_labels))}")
        print(f"")

        if use_wandb:
            log_pointcloud_to_wandb(points, gt_labels, name="Val_PCL/GroundTruth")
            log_pointcloud_to_wandb(points, pred_labels, name="Val_PCL/Prediction")

    writer.close()
    if use_wandb:
        wandb.finish()
        print("✅ Training complete.")

if __name__ == "__main__":
    main()