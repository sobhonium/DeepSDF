#!/usr/bin/env python
# coding: utf-8






# training two samples of ModelNet10
# 1) I'm reading the modeNet10 samples (via trimesh with .off extension)
# 2) I'm voxelizing them (binary) (via trimesh)
# 3) I'm convering binary girds to SDF grids (via scipy)
# 4) put the two samples in training and show the interpolation ...
# 5) show smooth transition from the first shape to the second shape





import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils import mesh_to_cubid_sdf


sys.path.append(os.path.abspath('..'))


# read all the setting from an outside file config.json ...
with open('../config.json', 'r') as file:
    setting = json.load(file)



# general desciption of the file run here...
print(setting["desc"]["general"])


# to put the mesh in a grid cube of size cube_size*cube_size*cube_size
cube_size = setting["cube_size"]
# the resolution of mesh to voxel
resolution = setting["resolution"]
# from where the inputs are selected
directory_path = Path('../dataset')
# latent_code length for deepSDF
code_len = setting["latent_code_leg"]
# optimization's learning rate
lr = setting["learning_rate"]
# batchsize
batch_size = setting["batch_size"]





# building the training samples
features = np.empty((0,5))

counter = 1
# List all files and directories in the specified directory
for sample_file_name in  directory_path.iterdir():

    sdf_voxel = mesh_to_cubid_sdf(sample_file_name, 
                            resolution=resolution, 
                            cube_size=50)


    code = [counter]*setting["latent_code_leg"]
    shape = sdf_voxel.shape
    
    # Generate the indices of all elements in the array
    i, j, k = np.indices(shape)
    
    # Flatten the indices and sdf_voxel values
    i,j,k = i.flatten(), j.flatten(),  k.flatten()
    sdf_values = sdf_voxel.flatten()
    
    # Stack the indices, code, and sdf_values
    code_array = np.full(i.shape, code[0])
    features1 = np.vstack((i, j, k, code_array, sdf_values)).T
    
    features = np.concatenate((features, features1), axis=0)
    
    counter+=1








device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = setting["device"]
print("============ Runnin on:", device,"="*10)





# features and labels extraction

labels = torch.tensor( features[:,4])#torch.tensor(features["labels"]).to(device)  # / (features["labels"].abs()).max())
features = features[:,:4]
print(f'labels are transfered to {setting["device"]}...')





# scaling the samples before training


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

features_scaler = StandardScaler()
# features_scaler = Pipeline([('scale',StandardScaler()),
#                 #  ('normalizing', QuantileTransformer(output_distribution='normal')),
#                 ])

# labels_scaler = Pipeline([('scale',StandardScaler()),
#                  ('normalizing', QuantileTransformer(output_distribution='normal')),
#                 ])

features = features_scaler.fit_transform(features)
features = torch.tensor(features).to(device) 
print(f'features are scaled and transered to {device}...')




joblib.dump(features_scaler, setting["featureScaler"])
print(f'{setting["featureScaler"]} is saved on the Disk for future...')




print(features_scaler.mean_, features_scaler.scale_)




# Combine inputs and labels into a TensorDataset
dataset = TensorDataset(features, labels)

train_size =  int(0.7 * len(dataset))  # 70% of the data for training
val_size = int(0.15 * len(dataset))    # 15% of the data for validation
test_size = int(0.15 * len(dataset))+0    # 15% of data for test

# print('diff sizes:   ',test_size, len(dataset) - train_size - val_size )
from torch.utils.data import random_split

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


# not a good one since I'm not using cross validtion but just for testing



# Create DataLoader instances for training, validation, and test sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)





from model.deepSDFmodel import DeepSDFModel4

code_len = 1
model = DeepSDFModel4(
              feature_len = 3,       # cordinate_dim=3 ihtout coef=3, 
              code_len=code_len,
              finetune_dim=0,
              hidden_dim=32,
              ).to(setting["device"]) 
print(f'sdf model is transfered to {device}...')



optimizer = torch.optim.Adam(model.parameters(), lr=lr)



loss = nn.MSELoss()



dataset_size = features.shape[0]



train_err = []
valid_err = []



#  Training Loop

epochs = setting["epochs"]

print(f'Training started on {device}...')
model.train()    
for epoch in range(epochs):
    train_loss = 0
    counter_train = 0
    for batch_idx, (x, y) in enumerate(train_dataloader):

        optimizer.zero_grad()
        y_hat = model(x.float()).float().to(device)
        ll = loss(y_hat.flatten(), y.float().flatten()).to(device)
        train_loss += ll.item()#*batch_size

        ll.backward()
        optimizer.step()
        counter_train+=1
          
    train_err.append(train_loss/counter_train)

    valid_loss=0
    counter_test = 0
    for x, y in test_dataloader:
        y_hat = model(x.float()).float().to(device)
        ll = loss(y_hat.flatten(), y.float().flatten()).to(device)
        valid_loss += ll.item()#*batch_size
        counter_test += 1
        
    valid_err.append(valid_loss/counter_test)
    
    torch.save(model.state_dict(), setting["DeepSDFModel"])   
    print("epoch= ", epoch, "/",epochs,"  Avg Training Error= ", train_loss/counter_train, flush=True)




maxx_ = 100
minn_ = 0
step = 1



import matplotlib.pyplot as plt

fig, axs = plt.subplots(figsize=(12, 8))

axs.plot(train_err[minn_:maxx_:step], label='Train Error')
axs.plot(valid_err[minn_:maxx_:step], label='Validation Error ')
axs.set_title('DeepSDF Error ')
axs.legend()
axs.set_xlabel('epoch')
axs.set_ylabel('Error')

plt.tight_layout()


plt.show()


