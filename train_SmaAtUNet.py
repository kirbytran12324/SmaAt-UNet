from typing import Optional

from models.SmaAt_UNet import SmaAt_UNet
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torchvision import transforms

from root import ROOT_DIR
import time
from tqdm import tqdm
from metric import iou
import os

from utils.dataset_precip import precipitation_maps_classification_h5


# Function to get the learning rate of the optimizer
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# Function to fit the model
# epochs: number of epochs to train for
# model: the model to train
# loss_func: the loss function to use
# opt: the optimizer to use
# train_dl: the DataLoader for the training data
# valid_dl: the DataLoader for the validation data
# dev: the device to use for training
# save_every: how often to save the model
# tensorboard: whether to use TensorBoard
# earlystopping: the number of epochs to wait before stopping training if validation loss does not improve
# lr_scheduler: the learning rate scheduler to use
def fit(
        epochs,
        model,
        loss_func,
        opt,
        train_dl,
        valid_dl,
        dev=None,
        save_every: Optional[int] = None,
        tensorboard: bool = False,
        earlystopping=None,
        lr_scheduler=None,
):
    # Initialize TensorBoard writer if tensorboard is True
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(comment=f"{model.__class__.__name__}")

    # Set device to cuda if available, else cpu
    if dev is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize variables for tracking time and best mIoU
    start_time = time.time()
    best_mIoU = -1.0
    earlystopping_counter = 0

    # Start training loop
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        # Set model to training mode
        model.train()
        train_loss = 0.0

        # Iterate over training data
        for _, (xb, yb) in enumerate(tqdm(train_dl, desc="Batches", leave=False)):
            # Compute loss
            loss = loss_func(model(xb.to(dev)), yb.to(dev))
            # Zero gradients
            opt.zero_grad()
            # Backpropagate
            loss.backward()
            # Update weights
            opt.step()
            # Update training loss
            train_loss += loss.item()

        # Compute average training loss
        train_loss /= len(train_dl)

        # Initialize validation loss and IoU metric
        val_loss = 0.0
        iou_metric = iou.IoU(21, normalized=False)

        # Set model to evaluation mode
        model.eval()

        # No gradient computation during validation
        with torch.no_grad():
            # Iterate over validation data
            for xb, yb in tqdm(valid_dl, desc="Validation", leave=False):
                # Compute predictions
                y_pred = model(xb.to(dev))
                # Compute loss
                loss = loss_func(y_pred, yb.to(dev))
                # Update validation loss
                val_loss += loss.item()

                # Compute mean IoU
                pred_class = torch.argmax(nn.functional.softmax(y_pred, dim=1), dim=1)
                iou_metric.add(pred_class, target=yb)

            # Compute average validation loss and mean IoU
            iou_class, mean_iou = iou_metric.value()
            val_loss /= len(valid_dl)

        # Save model if mean IoU is better than the best seen so far
        if mean_iou > best_mIoU:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "model": model,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "mIOU": mean_iou,
                },
                ROOT_DIR / "checkpoints" / f"best_mIoU_model_{model.__class__.__name__}.pt",
            )
            best_mIoU = mean_iou
            earlystopping_counter = 0

        else:
            earlystopping_counter += 1
            if earlystopping is not None and earlystopping_counter >= earlystopping:
                print(f"Stopping early --> mean IoU has not decreased over {earlystopping} epochs")
                break

        print(
            f"Epoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min,"
            f"Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f},",
            f"mIOU: {mean_iou:.10f},",
            f"lr: {get_lr(opt)},",
            f"Early stopping counter: {earlystopping_counter}/{earlystopping}" if earlystopping is not None else "",
        )

        if writer:
            # add to tensorboard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metric/mIOU", mean_iou, epoch)
            writer.add_scalar("Parameters/learning_rate", get_lr(opt), epoch)
        if save_every is not None and epoch % save_every == 0:
            # save model
            torch.save(
                {
                    "model": model,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "mIOU": mean_iou,
                },
                ROOT_DIR / "checkpoints" / f"model_{model.__class__.__name__}_epoch_{epoch}.pt",
            )
        if lr_scheduler is not None:
            lr_scheduler.step(mean_iou)


if __name__ == "__main__":
    # Set device to cuda if available, else cpu
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define dataset folder, batch size, learning rate, number of epochs, early stopping criteria, and save frequency
    in_file = "dataset/composed/train_test_2019-2020_input-length_12_img-ahead_6_rain-threshold_0.h5"
    num_input_images = 24
    img_to_predict = 5
    batch_size = 8
    learning_rate = 0.001
    epochs = 200
    earlystopping = 30
    save_every = 1

    # Define transformations for the dataset
    transformations = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

    # Load dataset for training and validation
    dataset_train = precipitation_maps_classification_h5(
        in_file=in_file,
        num_input_images=num_input_images,
        img_to_predict=img_to_predict,
        train=True,
    )

    dataset_val = precipitation_maps_classification_h5(
        in_file=in_file,
        num_input_images=num_input_images,
        img_to_predict=img_to_predict,
        train=False,
    )

    # Create DataLoaders for training and validation datasets
    train_dl = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    valid_dl = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Initialize SmaAt-UNet model
    model = SmaAt_UNet(n_channels=3, n_classes=21)
    # Move model to device
    model.to(dev)

    # Define optimizer and loss function
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss().to(dev)

    # Define learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=4)

    # Start training
    fit(
        epochs=epochs,
        model=model,
        loss_func=loss_func,
        opt=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
        dev=dev,
        save_every=save_every,
        tensorboard=True,
        earlystopping=earlystopping,
        lr_scheduler=lr_scheduler,
    )
