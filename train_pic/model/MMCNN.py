# Definition of the 8 sub-models : 
from torch import nn
from torch.nn import functional as F

class CNN_M1(nn.Module):
    def __init__(self):
        super(CNN_M1, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,64,3, padding =(1,1))
        self.conv4 = nn.Conv2d(64,128,3, padding =(1,1))
        self.conv5 = nn.Conv2d(128,1,3, padding =(1,1))
        self.dropout = nn.Dropout(0.35)
    def forward(self, x):
        # print('in CNN: ', x.shape)
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        # print('in CNN output: ', x.shape)
        return x
    

class CNN_M2(nn.Module):
    def __init__(self):
        super(CNN_M2, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,64,3, padding =(1,1))
        self.conv4 = nn.Conv2d(64,128,3, padding =(1,1))
        self.conv5 = nn.Conv2d(128,1,3, padding =(1,1))
        self.dropout = nn.Dropout(0.35)
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        return x
    
class CNN_M3(nn.Module):
    def __init__(self):
        super(CNN_M3, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,64,3, padding =(1,1))
        self.conv4 = nn.Conv2d(64,128,3, padding =(1,1))
        self.conv5 = nn.Conv2d(128,1,3, padding =(1,1))
        self.dropout = nn.Dropout(0.35)
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        return x
    
class CNN_M4(nn.Module):
    def __init__(self):
        super(CNN_M4, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,64,3, padding =(1,1))
        self.conv4 = nn.Conv2d(64,128,3, padding =(1,1))
        self.conv5 = nn.Conv2d(128,1,3, padding =(1,1))
        self.dropout = nn.Dropout(0.35)
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        return x
    
class CNN_M5(nn.Module):
    def __init__(self):
        super(CNN_M5, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,64,3, padding =(1,1))
        self.conv4 = nn.Conv2d(64,128,3, padding =(1,1))
        self.conv5 = nn.Conv2d(128,1,3, padding =(1,1))
        self.dropout = nn.Dropout(0.35)
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        return x
    
class CNN_M6(nn.Module):
    def __init__(self):
        super(CNN_M6, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,64,3, padding =(1,1))
        self.conv4 = nn.Conv2d(64,128,3, padding =(1,1))
        self.conv5 = nn.Conv2d(128,1,3, padding =(1,1))
        self.dropout = nn.Dropout(0.35)
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        return x
    
class CNN_M7(nn.Module):
    def __init__(self):
        super(CNN_M7, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,64,3, padding =(1,1))
        self.conv4 = nn.Conv2d(64,128,3, padding =(1,1))
        self.conv5 = nn.Conv2d(128,1,3, padding =(1,1))
        self.dropout = nn.Dropout(0.35)
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        return x
    
class CNN_M8(nn.Module):
    def __init__(self):
        super(CNN_M8, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,64,3, padding =(1,1))
        self.conv4 = nn.Conv2d(64,128,3, padding =(1,1))
        self.conv5 = nn.Conv2d(128,1,3, padding =(1,1))
        self.dropout = nn.Dropout(0.35)     
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        return x
    
# Definition of the W attention module :
    
class CNN_W(nn.Module):
    def __init__(self):
        super(CNN_W, self).__init__()
        self.conv1 = nn.Conv2d(9,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,8,3, padding =(1,1))
        self.dropout = nn.Dropout(0.35)  
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.softmax(self.conv3(x)))
        return x