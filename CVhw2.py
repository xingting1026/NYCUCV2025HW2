import os
import json
import csv
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(
        f"GPU Memory: {
            torch.cuda.get_device_properties(0).total_memory /
            1024 /
            1024 /
            1024:.2f} GB")
else:
    device = torch.device("cpu")
    logger.info("CUDA is not available. Using CPU instead.")


# Dataset paths
DATA_ROOT = "nycu-hw2-data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
VALID_DIR = os.path.join(DATA_ROOT, "valid")
TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
VALID_JSON = os.path.join(DATA_ROOT, "valid.json")

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DigitDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            annotation_file=None,
            transform=None,
            is_test=False):
        self.root = root
        self.transform = transform
        self.is_test = is_test
        self.image_files = [f for f in os.listdir(
            root) if f.endswith(".jpg") or f.endswith(".png")]
        self.image_files.sort(
            key=lambda x: int(
                x.split(".")[0]))  # Sort by image id

        # Annotations (for training/validation)
        self.annotations = []
        if not is_test and annotation_file:
            with open(annotation_file, "r") as f:
                coco_data = json.load(f)

            # Create image_id to annotation mapping
            img_id_to_annotations = {}
            for ann in coco_data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in img_id_to_annotations:
                    img_id_to_annotations[img_id] = []
                img_id_to_annotations[img_id].append(ann)

            # Store category information
            self.categories = {
                cat["id"]: cat["name"] for cat in coco_data["categories"]
            }

            # Create a list of annotations for each image
            for img in coco_data["images"]:
                img_id = img["id"]
                if img_id in img_id_to_annotations:
                    self.annotations.append(img_id_to_annotations[img_id])
                else:
                    # No annotations for this image
                    self.annotations.append([])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        # Get image id
        img_id = int(self.image_files[idx].split(".")[0])

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        # For test set, just return the image and id
        if self.is_test:
            return img, img_id

        # For training/validation, prepare target
        target = {}
        boxes = []
        labels = []

        # Get annotations for this image
        for ann in self.annotations[idx]:
            # Bounding box: [x_min, y_min, width, height] -> [x_min, y_min,
            # x_max, y_max]
            box = ann["bbox"]
            x_min, y_min, width, height = box
            x_max = x_min + width
            y_max = y_min + height

            # Add box and label
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])

        # Convert to tensors
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Empty annotations
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0), dtype=torch.int64)

        return img, target, img_id


# Define collate function at module level to avoid serialization issues
def collate_fn(batch):
    return tuple(zip(*batch))


def create_data_loaders(
        train_dataset,
        valid_dataset,
        test_dataset,
        batch_size=4):
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Changed from 2 to 0 to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Changed from 2 to 0
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Changed from 2 to 0
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, valid_loader, test_loader


def create_faster_rcnn_model(num_classes):
    """
    Create a Faster R-CNN model with ResNet50-FPN backbone
    """
    # Load pre-trained model with FPN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT")

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Customize anchor sizes and aspect ratios for digit detection
    # FPN has 5 feature maps, so we need 5 sizes and 5 aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=(
            (16,),
            (32,),
            (64,),
            (128,),
            (256,),
        ),  # Different sizes for each feature map
        # Same aspect ratios for all feature maps
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model.rpn.anchor_generator = anchor_generator

    # Adjust detection parameters
    model.roi_heads.score_thresh = 0.5  # Score threshold for detections
    model.roi_heads.nms_thresh = 0.3  # NMS threshold for overlapping boxes
    model.roi_heads.detections_per_img = 100  # Max detections per image

    return model


def train_one_epoch(model, data_loader, optimizer, epoch, device):
    """
    Train the model for one epoch
    """
    model.train()

    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")

    for images, targets, _ in progress_bar:
        # Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Check if loss is valid
        if not torch.isfinite(losses):
            logger.error(f"Loss is {losses}, stopping training")
            logger.error(f"Loss dict: {loss_dict}")
            continue

        # Backward pass and optimize
        optimizer.zero_grad()
        losses.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update progress bar
        running_loss += losses.item()
        if "loss_classifier" in loss_dict:
            running_loss_classifier += loss_dict["loss_classifier"].item()
        if "loss_box_reg" in loss_dict:
            running_loss_box_reg += loss_dict["loss_box_reg"].item()
        if "loss_objectness" in loss_dict:
            running_loss_objectness += loss_dict["loss_objectness"].item()
        if "loss_rpn_box_reg" in loss_dict:
            running_loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()

        progress_bar.set_postfix(
            {
                "loss": running_loss / (progress_bar.n + 1),
                "cls_loss": running_loss_classifier / (progress_bar.n + 1),
                "box_loss": running_loss_box_reg / (progress_bar.n + 1),
            }
        )

    # Calculate average losses
    avg_loss = running_loss / len(data_loader)
    avg_loss_classifier = running_loss_classifier / len(data_loader)
    avg_loss_box_reg = running_loss_box_reg / len(data_loader)
    avg_loss_objectness = running_loss_objectness / len(data_loader)
    avg_loss_rpn_box_reg = running_loss_rpn_box_reg / len(data_loader)

    # Log losses
    logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}")
    logger.info(f"  Classifier Loss: {avg_loss_classifier:.4f}")
    logger.info(f"  Box Reg Loss: {avg_loss_box_reg:.4f}")
    logger.info(f"  Objectness Loss: {avg_loss_objectness:.4f}")
    logger.info(f"  RPN Box Reg Loss: {avg_loss_rpn_box_reg:.4f}")

    return avg_loss


def validate(model, data_loader, device):
    """
    Validate the model
    """
    model.train()  # 暫時設為訓練模式，以取得損失值

    running_loss = 0.0

    with torch.no_grad():
        for images, targets, _ in tqdm(data_loader, desc="Validation"):
            # 移動資料到裝置
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            # 前向傳播
            loss_dict = model(images, targets)

            # 確保 loss_dict 是字典而非列表
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
                running_loss += losses.item()
            else:
                logger.warning(
                    "Model returned non-dict output during validation, "
                    "skipping batch"
                )
                continue

    # 計算平均損失
    avg_loss = running_loss / len(data_loader)

    # 記錄驗證損失
    logger.info(f"Validation Loss: {avg_loss:.4f}")

    # 切換回評估模式
    model.eval()

    return avg_loss


def train_model(model, train_loader, valid_loader, num_epochs=10, lr=0.001):
    """
    Train the model for multiple epochs
    """
    model.to(device)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, epoch, device)

        # Update learning rate
        lr_scheduler.step()

        # Validate the model
        valid_loss = validate(model, valid_loader, device)

        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(
                    OUTPUT_DIR,
                    "best_model.pth"))
            logger.info(
                f"Saved best model with validation loss: {
                    best_valid_loss:.4f}")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            },
            os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch + 1}.pth"),
        )

        logger.info(f"Saved checkpoint for epoch {epoch + 1}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    logger.info("Training completed and final model saved.")


def predict_task1(model, test_loader):
    """
    Predict bounding boxes and classes for Task 1
    """
    logger.info("Starting Task 1 prediction...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Predicting Task 1"):
            # 移動圖像到裝置
            images = list(image.to(device) for image in images)

            # 前向傳播
            outputs = model(images)

            # 處理輸出結果
            for i, output in enumerate(outputs):
                # 檢查 image_ids[i] 是否為張量
                if torch.is_tensor(image_ids[i]):
                    img_id = image_ids[i].item()
                else:
                    img_id = image_ids[i]  # 已經是整數，不需呼叫 .item()

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.5:  # 置信度閾值
                        # 將邊界框從 [x_min, y_min, x_max, y_max] 轉換為 [x_min, y_min,
                        # width, height]
                        x_min, y_min, x_max, y_max = box
                        width = x_max - x_min
                        height = y_max - y_min

                        predictions.append(
                            {
                                "image_id": img_id,
                                "bbox": [
                                    float(x_min),
                                    float(y_min),
                                    float(width),
                                    float(height),
                                ],
                                "score": float(score),
                                "category_id": int(label),
                            }
                        )

    # 將預測結果儲存為 COCO 格式
    logger.info(f"Writing {len(predictions)} predictions to file...")
    with open(os.path.join(OUTPUT_DIR, "pred.json"), "w") as f:
        json.dump(predictions, f)

    logger.info("Task 1 prediction completed.")
    return predictions


def predict_task2(predictions, test_image_ids):
    """
    Predict digit numbers for Task 2 based on Task 1 predictions
    """
    logger.info("Starting Task 2 prediction...")

    # Group predictions by image_id
    img_id_to_predictions = {}
    for pred in predictions:
        img_id = pred["image_id"]
        if img_id not in img_id_to_predictions:
            img_id_to_predictions[img_id] = []
        img_id_to_predictions[img_id].append(pred)

    # Initialize list to store results for all test images
    results = []

    # Process each test image
    for img_id in test_image_ids:
        # If no predictions for this image, predict -1
        if (
            img_id not in img_id_to_predictions
            or len(img_id_to_predictions[img_id]) == 0
        ):
            results.append([img_id, -1])
            continue

        # Sort predictions by x-coordinate of the bounding box
        sorted_preds = sorted(
            img_id_to_predictions[img_id],
            key=lambda x: x["bbox"][0])

        # Extract digits and combine them into a number
        digits = [
            str(pred["category_id"] - 1) for pred in sorted_preds
        ]  # Subtract 1 because COCO categories start from 1
        number = int("".join(digits))

        results.append([img_id, number])

    # Write results to CSV file
    logger.info(f"Writing {len(results)} results to CSV file...")
    with open(os.path.join(OUTPUT_DIR, "pred.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "pred_label"])
        writer.writerows(results)

    logger.info("Task 2 prediction completed.")
    return results


def main():
    # Set random seed for reproducibility
    set_seed()

    # Define transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    # Create datasets
    train_dataset = DigitDataset(
        root=TRAIN_DIR, annotation_file=TRAIN_JSON, transform=transform
    )

    valid_dataset = DigitDataset(
        root=VALID_DIR, annotation_file=VALID_JSON, transform=transform
    )

    test_dataset = DigitDataset(
        root=TEST_DIR,
        transform=transform,
        is_test=True)

    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset, batch_size=4
    )

    # Number of classes (10 digits + background)
    num_classes = 11  # 0-9 digits + background (0)

    # Training parameters - reduce for CPU, increase for GPU
    if torch.cuda.is_available():
        num_epochs = 8
        learning_rate = 0.001
    else:
        num_epochs = 5
        learning_rate = 0.001

    # Check if a trained model exists, otherwise train a new one
    model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if os.path.exists(model_path):
        logger.info(f"Loading pre-trained model from {model_path}")
        model = create_faster_rcnn_model(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logger.info("No pre-trained model found. Training a new model...")
        model = create_faster_rcnn_model(num_classes)
        train_model(
            model,
            train_loader,
            valid_loader,
            num_epochs,
            learning_rate)

    # Move model to device
    model.to(device)

    # Make predictions for Task 1
    predictions = predict_task1(model, test_loader)

    # Get all test image IDs
    test_image_ids = sorted([
        int(f.split(".")[0])
        for f in os.listdir(TEST_DIR)
        if f.endswith(".jpg") or f.endswith(".png")
    ])

    # Make predictions for Task 2
    predict_task2(predictions, test_image_ids)

    logger.info("All tasks completed successfully!")


if __name__ == "__main__":
    main()
