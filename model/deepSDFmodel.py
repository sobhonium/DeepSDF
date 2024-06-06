import torch.nn as nn
import torch.nn.functional as F
import torch

class MyBatchNormalizationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Define any attributes here

    def forward(self, x):
        return (x-x.mean())/torch.sqrt(x.var())
        # Perform the required processing on x and return it


# In[ ]:





class DeepSDFModel4(nn.Module):
    def __init__(self, code_len=6, feature_len=6, finetune_dim=2, hidden_dim=40):
        super(DeepSDFModel4, self).__init__()
        self.is_training = True
        self.feature_len = feature_len
        self.code_len = code_len
        self.finetune_dim = finetune_dim
        
        input_leng = self.feature_len + self.code_len + self.finetune_dim
        self.func1 = nn.Linear(input_leng, hidden_dim)
        self.bn1 = MyBatchNormalizationLayer()  # Batch Normalization layer
        
        self.func2 = nn.Linear(hidden_dim + input_leng, hidden_dim)
        self.bn2 = MyBatchNormalizationLayer()  # Batch Normalization layer
        
        self.func3 = nn.Linear(hidden_dim + input_leng, hidden_dim)
        self.bn3 = MyBatchNormalizationLayer()  # Batch Normalization layer
        
        self.func4 = nn.Linear(hidden_dim + input_leng, hidden_dim)
        self.bn4 = MyBatchNormalizationLayer()  # Batch Normalization layer
        
        self.func7 = nn.Linear(hidden_dim + input_leng , hidden_dim)
        self.bn7 = MyBatchNormalizationLayer()  # Batch Normalization layer
        
        self.func5 = nn.Linear(hidden_dim + input_leng , hidden_dim//2)
        self.bn5 = MyBatchNormalizationLayer() # Batch Normalization layer
        
        self.func6 = nn.Linear(hidden_dim//2, 1)
        
        self.outp = nn.Tanh()
        
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)  # Common dropout layer
        
    def forward(self, x):
        
        x_backup =  x.clone()
        x = self.func1(x.float())
        x = self.leakyReLU(x)
     
        x = torch.cat((x, x_backup), dim=1)
        x = self.bn2(x)  # Apply Batch Normalization
        x = self.func2(x)
        x = self.leakyReLU(x)
        
        x = torch.cat((x, x_backup), dim=1)
        x = self.bn3(x)  # Apply Batch Normalization
        x = self.func3(x)
        x = self.leakyReLU(x)
        
        x = torch.cat((x, x_backup), dim=1)
        x = self.bn4(x)  # Apply Batch Normalization
        x = self.func4(x)
        x = self.leakyReLU(x)
       
        x = torch.cat((x, x_backup), dim=1)
        x = self.bn7(x)  # Apply Batch Normalization
        x = self.func7(x)
        x = self.leakyReLU(x)
        
        x = torch.cat((x, x_backup), dim=1)
        x = self.bn5(x)  # Apply Batch Normalization
        x = self.func5(x)
        x = self.leakyReLU(x)
       

        x = self.func6(x)
        
        return x
        # return self.outp(x) 

