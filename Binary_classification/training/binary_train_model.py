import time
import tqdm

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import wandb

from training_sub import DataSet, EarlyStopping


def augment_data(train_data, augment_horizontal=False, augment_vertical=False, augment_velocity_axis=False):
    horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
    vertical_flip = transforms.RandomVerticalFlip(p=1.0)
    
    augmented_data = []
    for img in train_data:
        augmented_data.append(img)
        #左右反転
        if augment_horizontal:
            augmented_data.append(horizontal_flip(img))
        #上下反転
        if augment_vertical:
            augmented_data.append(vertical_flip(img))
        #速度反転
        if augment_velocity_axis:
            augmented_data.append(torch.flip(img, dims=[0]))
    
    train_data = torch.stack(augmented_data)
    return train_data


def train_model(model, criterion, optimizer, num_epochs, args, device, run, 
                augment_horizontal   =False, 
                augment_vertical     =False, 
                augment_velocity_axis=False):
    #weight_pass = args.savedir_path + "/model_parameters" + f"/model_parameter_{args.wandb_name}.pth"
    early_stopping = EarlyStopping(patience=15, verbose=True, path=args.savedir_path + "/model_parameter.pth")

    data = np.load(args.training_validation_path)
    # data = np.memmap(args.training_validation_path, dtype=np.float32, mode="r", shape=(11676, 120, 112, 112))
    data = torch.from_numpy(data).float()
    label = [0] * len(data)
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, label, test_size=0.2, random_state=42, stratify=label
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        val_data, val_labels, test_size=0.25, random_state=42, stratify=val_labels
    )

    train_data = augment_data(train_data, augment_horizontal, augment_vertical, augment_velocity_axis)
    train_labels     = [0] * len(train_data)

    train_dataset    = DataSet(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_mini_batch, shuffle=True)
    val_dataset      = DataSet(val_data, val_labels)
    val_dataloader   = DataLoader(val_dataset, batch_size=args.val_mini_batch, shuffle=False)
    dataloader_dic   = {"train": train_dataloader, "val": val_dataloader}


    train_dataset = DataSet(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_mini_batch, shuffle=True)
    val_dataset = DataSet(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_mini_batch, shuffle=False)
    dataloader_dic = {"train": train_dataloader, "val": val_dataloader}

    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    start = time.time()
    for epoch in range(args.num_epoch):
        train_loss_num = 0
        val_loss_num = 0

        for phase in ["train", "val"]:
            dataloader = dataloader_dic[phase]
            if phase == "train":
                model.train()  # モデルを訓練モードに
            else:
                model.eval()

            for images, labels in tqdm.tqdm(dataloader):
                images = images.view(-1, 1, 30, 100, 100)  # バッチサイズを維持したままチャンネル数を1に設定
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):

                    # モデルの出力を計算する
                    output = model(images.clone().to(device))

                    # 損失を計算する
                    loss = criterion(output.to("cpu"), images)
                    weighted_loss = torch.mean(loss)

                    # パラメータの更新
                    if phase == "train":
                        weighted_loss.backward()
                        optimizer.step()
                        train_loss_num += weighted_loss.item()
                    else:
                        val_loss_num += weighted_loss.item()

            if phase == "train":
                train_loss_list.append(train_loss_num)
            else:
                val_loss_list.append(val_loss_num)
                
        wandb.log({"train loss": train_loss_num, "validation loss": val_loss_num, "epoch":  epoch})
        if val_loss_num < best_val_loss:
            best_val_loss = val_loss_num
            wandb.log({"best validation loss": best_val_loss, "epoch":  epoch})
        
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, val_loss_num))

        early_stopping(val_loss_num, model)
        if early_stopping.early_stop:
            print("Early_Stopping")
            break

    #train_loss_path = args.savedir_path + "/loss_log" + f"/train_loss_{args.wandb_name}.npy"
    #val_loss_path = args.savedir_path + "/loss_log" + f"/val_loss_{args.wandb_name}.npy"


    #np.save(train_loss_path, train_loss_list)
    #np.save(val_loss_path, val_loss_list)

    print((time.time() - start) / 60)
