import sys
sys.path.append("/work/u8083200/Thesis/SOTA/rawhdr")

from module import rawhdr_model as module
import dataset
import config
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import matplotlib.pyplot as plt
import random
import utils

if __name__ == "__main__":
    
    print("Load Config")
    data_root = config.HDR_ROOT
    batch_size = config.BATCH_SIZE
    epochs = config.EPOCH
    aug = config.AUG
    
    # weight_name = config.SPATIAL_WEIGHT_NAME
    weight_name = config.WEIGHT_NAME
    result_save_path = config.RESULT_SAVE_PATH
    weight_save_path = config.WEIGHT_SAVE_PATH
    learning_rate = config.LEARNING_RATE
    device = config.DEVICE
    
    result_root = os.path.join(result_save_path, weight_name)
    
    print("\nWEIGHT_NAME: {}".format(weight_name))
    print("AUG: {}".format(aug))
    
    # ----- Load Data -----
    print("Load Data...")
    hdr_path = []
    ldr_path = []
    
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
        
    if not os.path.exists(result_root):
        os.mkdir(result_root)
        os.mkdir(os.path.join(result_root, "train"))
        os.mkdir(os.path.join(result_root, "val"))
    
    if not os.path.exists(weight_save_path):
        os.mkdir(weight_save_path)
    
    for root, dirs, files in os.walk(data_root):
        files.sort()
        for file in files:
            if(".hdr" in file or ".tif" in file):
                hdr_path.append(os.path.join(root, file))
                continue
    
    # Validation data
    val_path = []
    val_root = "/work/u8083200/Thesis/datasets/HDR-Real"
    for root, dirs, files in os.walk(val_root):
        files.sort()
        for file in files:
            if("gt.hdr" in file):
                val_path.append(os.path.join(root, file))
                continue
    
    random.seed(2454)
    Train_HDR = hdr_path
    Val_HDR = val_path
    
    train_dataloader = DataLoader(dataset.dataset(Train_HDR, stage=4, image_size=512, aug=aug), shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(dataset.dataset(Val_HDR, stage=4, image_size=512, aug=aug), shuffle=True, batch_size=batch_size)
    
    print("Set Model")
    model = module.rawhdr_model()
    model.apply(module.weights_init)
    model.to(device)
    
    config.set_random_seed(2454)
    
    if os.path.isfile(os.path.join(os.path.join(weight_save_path, weight_name), "best.pth")):
        max_epoch = 0 # Setting LR
        for root, dirs, files in os.walk(os.path.join(weight_save_path, weight_name)):
            if("best.pth" not in files):
                break
            files.remove ("best.pth")
            for file in files:
                if int(file[:-4]) > max_epoch:
                    max_epoch = int(file[:-4])
                    
        model.load_state_dict(torch.load(os.path.join(os.path.join(weight_save_path, weight_name),
                                                      str(max_epoch) + ".pth")))
        learning_rate = learning_rate * pow(0.99, max_epoch)
        weight_name = weight_name + "_conti"
        
        result_root = os.path.join(result_save_path, weight_name)
        
        if not os.path.exists(result_root):
            os.mkdir(result_root)
            os.mkdir(os.path.join(result_root, "train"))
            os.mkdir(os.path.join(result_root, "val"))
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-1)
    
    writer = SummaryWriter("runs/rawhdr/" + weight_name + "_log_" + time.ctime(time.time()), 
                           comment=weight_name)
    
    loss_dir = {}
    
    loss_weight = config.loss_weight
    fig, ax = plt.subplots()
    bars = ax.bar(*zip(*loss_weight.items()))
    ax.bar_label(bars)
    writer.add_figure("Loss Weight", fig)
    
    if not os.path.exists(os.path.join(weight_save_path, weight_name)):
        os.mkdir(os.path.join(weight_save_path, weight_name))
    
    
    print("Model Name: {}".format(weight_name))
    # Start Training Section
    train_iteration = 0
    val_iteration = 0
    test_iteration = 0
    min_loss = 9999
    for epoch in range(epochs):
        tqdm_bar = tqdm(train_dataloader, desc=f"Train Epoch {epoch+1} / {epochs}",
                        total=int(len(train_dataloader)))
        
        # Init Loss value, epoch sample count
        loss_dir = {key: 0 for key in loss_dir}
        
        epoch_sample = 0 
        # Training
        for batch_idx, imgs in enumerate(tqdm_bar):
            model.train()
            source = imgs["source"].to(device)
            target = imgs["target"].to(device)
            target_mask = {"over": imgs["over"].to(device).to(torch.float32),
                           "under": imgs["under"].to(device).to(torch.float32)}

            source = source.to(torch.float32)
            target = target.to(torch.float32)
            
            optimizer.zero_grad()
            
            # Model Run
            pd_output, pd_mask = model(source)
            
            predict_dict = {"output": pd_output,
                            "mask": pd_mask}
            
            target_dict = {"output": target,
                           "mask": target_mask}
            
            loss, loss_dict = utils.loss(predict_dict, target_dict,
                                         loss_weight)
            
            loss.backward()
            optimizer.step()
            
            epoch_sample += 1
            
            for key, value in loss_dict.items():
                if batch_idx == 0:
                    loss_dir[key] = loss_dict[key].item()
                    continue
                
                loss_dir[key] += loss_dict[key].item()
            
            max_value = torch.amax(target, dim=(1, 2, 3))
            max_value = torch.log(torch.add(10 * max_value, 1)) / torch.log(torch.tensor(1 + 10))
            if (train_iteration % 999 == 0):
                writer.add_image("Train_Source/Source_LDR", dataset.eval_image(source), train_iteration)
                
                writer.add_image("Train_Out_texture/Target", dataset.eval_image(target, file_name=result_root + "/train/" + str(train_iteration) + "_gt.hdr"), train_iteration)
                writer.add_image("Train_Out_texture/Predict_output", dataset.eval_image(pd_output, file_name=result_root + "/train/" + str(train_iteration) + "_pd.hdr"), train_iteration)
                
                writer.add_image("Train_Out_texture_log/Target_log", dataset.eval_image(target, log=True, max_value=max_value), train_iteration)
                writer.add_image("Train_Out_texture_log/Predict_output_log", dataset.eval_image(pd_output, log=True, max_value=max_value), train_iteration)
                
                writer.add_image("Train_Out_structure/Target_over", dataset.eval_image(target_mask["over"]), train_iteration)
                writer.add_image("Train_Out_structure/Predict_over", dataset.eval_image(pd_mask["over"]), train_iteration)
                
                writer.add_image("Train_Out_structure/Target_under", dataset.eval_image(target_mask["under"]), train_iteration)
                writer.add_image("Train_Out_structure/Predict_under", dataset.eval_image(pd_mask["under"]), train_iteration)
                writer.flush()
            train_iteration += 1
            
            loss_saving = {key: value / epoch_sample for key, value in loss_dir.items()}
            tqdm_bar.set_postfix(loss_saving)
        
        scheduler.step()
        for key, value in loss_dir.items():
            writer.add_scalar("Train/" + key, loss_dir[key] / epoch_sample, epoch)

        # Val
        freq = 1
        if (epoch + 1) % freq == 0:
            tqdm_bar = tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1} / {epochs}',
                            total=int(len(val_dataloader)))
            
            loss_dir = {key: 0 for key in loss_dir}
            
            epoch_sample = 0 
            
            for batch_idx, imgs in enumerate(tqdm_bar):
                model.eval()
                
                with torch.no_grad():
                    source = imgs["source"].to(device)
                    target = imgs["target"].to(device)
                    target_mask = {"over": imgs["over"].to(device).to(torch.float32),
                                "under": imgs["under"].to(device).to(torch.float32)}
                    
                    source = source.to(torch.float32)
                    target = target.to(torch.float32)
                    
                    optimizer.zero_grad()
                    
                    # Model Run
                    pd_output, pd_mask = model(source)
                    
                    predict_dict = {"output": pd_output,
                                    "mask": pd_mask}
                    
                    target_dict = {"output": target,
                                "mask": target_mask}
                    
                    loss, loss_dict = utils.loss(predict_dict, target_dict,
                                                loss_weight)
                    epoch_sample += 1
                    
                    for key, value in loss_dict.items():
                        if batch_idx == 0:
                            loss_dir[key] = loss_dict[key].item()
                            continue
                
                    loss_dir[key] += loss_dict[key].item()
                    
                    max_value = torch.amax(target, dim=(1, 2, 3))
                    max_value = torch.log(torch.add(10 * max_value, 1)) / torch.log(torch.tensor(1 + 10))
                    if (val_iteration % 59 == 0):                      
                        writer.add_image("Validation_Source/Source_LDR", dataset.eval_image(source), val_iteration)
                                
                        writer.add_image("Validation_Source/Source_LDR", dataset.eval_image(source), val_iteration)
                        
                        writer.add_image("Validation_Out_texture/Target", dataset.eval_image(target, file_name=result_root + "/val/" + str(val_iteration) + "_gt.hdr"), val_iteration)
                        writer.add_image("Validation_Out_texture/Predict_output", dataset.eval_image(pd_output, file_name=result_root + "/val/" + str(val_iteration) + "_pd.hdr"), val_iteration)
                        
                        writer.add_image("Validation_Out_texture_log/Target_log", dataset.eval_image(target, log=True, max_value=max_value), val_iteration)
                        writer.add_image("Validation_Out_texture_log/Predict_output_log", dataset.eval_image(pd_output, log=True, max_value=max_value), val_iteration)
                        
                        writer.add_image("Validation_Out_structure/Target_over", dataset.eval_image(target_mask["over"]), val_iteration)
                        writer.add_image("Validation_Out_structure/Predict_over", dataset.eval_image(pd_mask["over"]), val_iteration)
                        
                        writer.add_image("Validation_Out_structure/Target_under", dataset.eval_image(target_mask["under"]), val_iteration)
                        writer.add_image("Validation_Out_structure/Predict_under", dataset.eval_image(pd_mask["under"]), val_iteration)
                        
                        writer.flush()
                    val_iteration += 1
        
                    loss_saving = {key: value / epoch_sample for key, value in loss_dir.items()}
                    tqdm_bar.set_postfix(loss_saving)
                    
            for key, value in loss_dir.items():
                writer.add_scalar("Val/" + key, loss_dir[key] / epoch_sample, epoch)
            
        if((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), os.path.join(os.path.join(weight_save_path, weight_name), str(epoch+1)+'.pth'))
        if(loss_dir["loss"] / epoch_sample < min_loss):
            min_loss = loss_dir["loss"] / epoch_sample
            torch.save(model.state_dict(), os.path.join(os.path.join(weight_save_path, weight_name), 'best.pth'))
            
    print("Finish Training :)")