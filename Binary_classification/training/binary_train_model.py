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
    early_stopping = EarlyStopping(patience=20, verbose=True, path=args.savedir_path + "/model_parameter.pth")

    train_data = np.load(args.training_path)
    train_labels = np.load(args.training_labels_path)
    val_data = np.load(args.validation_path)
    val_labels = np.load(args.validation_labels_path)

    train_data = torch.from_numpy(train_data).float()
    train_labels = torch.from_numpy(train_labels).float()
    val_data = torch.from_numpy(val_data).float()
    val_labels = torch.from_numpy(val_labels).float()

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

    # train_loss_list = []
    # val_loss_list = []
    best_val_loss = float('inf')
    start = time.time()
    for epoch in range(args.num_epoch):
        train_loss_num = 0
        val_loss_num = 0
        
        # 精度計算のためのカウンター
        val_correct_preds = 0
        val_total_samples = 0
        val_true_positives = 0
        val_actual_positives = 0 # (TP + FN)
        val_predicted_positives = 0

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

                    if phase == "val":
                        predicted = (output > 0.5).float()
                        
                        # 1. 精度 (Accuracy) の計算
                        val_correct_preds += (predicted == labels).sum().item()
                        val_total_samples += labels.size(0)
                        
                        # 2. Recallの計算
                        # a. True Positives (TP): predicted=1 かつ actual=1
                        val_true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                        
                        # b. Actual Positives (TP + FN): actual=1 (正解ラベルが1の総数)
                        val_actual_positives += (labels == 1).sum().item()
    
                        # 3. Precision用 【追加】 (分母: 予測した正例数)
                        val_predicted_positives += (predicted == 1).sum().item()
                    
                    # パラメータの更新
                    if phase == "train":
                        weighted_loss.backward()
                        optimizer.step()
                        train_loss_num += weighted_loss.item()
                    else:
                        val_loss_num += weighted_loss.item()
                        
            # エポック終了後の検証精度の計算
            val_accuracy = val_correct_preds / val_total_samples if val_total_samples > 0 else 0.0
            # 検証再現率 (Recall) の計算
            val_recall = val_true_positives / val_actual_positives if val_actual_positives > 0 else 0.0
            # Precision = TP / (TP + FP) = TP / Predicted Positives 【追加】
            val_precision = val_true_positives / val_predicted_positives if val_predicted_positives > 0 else 0.0

            # if phase == "train":
            #     train_loss_list.append(train_loss_num)
            # else:
            #     val_loss_list.append(val_loss_num)
                
        wandb.log({"train loss": train_loss_num, "validation loss": val_loss_num, "epoch":  epoch})
        wandb.log({"validation accuracy": val_accuracy, "validation recall": val_recall, "validation precision": val_precision, "epoch": epoch})
        
        if val_loss_num < best_val_loss:
            best_val_loss = val_loss_num
            wandb.log({"best validation loss": best_val_loss, "epoch":  epoch})
        
        print("Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Recall: {:.4}, Precison: {:.4}".format(epoch + 1, num_epochs, val_loss_num, val_accuracy, val_recall, val_precision))

        early_stopping(val_loss_num, model)
        if early_stopping.early_stop:
            print("Early_Stopping")
            break

    #train_loss_path = args.savedir_path + "/loss_log" + f"/train_loss_{args.wandb_name}.npy"
    #val_loss_path = args.savedir_path + "/loss_log" + f"/val_loss_{args.wandb_name}.npy"


    #np.save(train_loss_path, train_loss_list)
    #np.save(val_loss_path, val_loss_list)

    print((time.time() - start) / 60)
