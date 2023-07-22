from __future__ import print_function
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

dropout_value = 0.01

class Net_assign9(nn.Module):
    def __init__(self):
        super(Net_assign9, self).__init__()
        # Input Block
        # CONVOLUTION BLOCK 1 input 32/1/1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),

            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32/3/1

        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, groups=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 32/5/1

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=64, 
                kernel_size=(3,3), 
                padding=2, 
                stride=2, 
                dilation=2, 
                bias=True),
        ) # output_size = 16/7/2

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, groups=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 16/11/2

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=128, groups=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 16/15/2

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, 
                      out_channels=64, 
                      kernel_size=(3,3), 
                      padding=2, 
                      dilation=2,
                      stride=2),
        ) # output_size = 8/19/4

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, groups=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 8/24/4
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, groups = 64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 8/32/4

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, groups=64,  out_channels=64, kernel_size=(3, 3), padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 6/40/4
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1/64

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)        
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



# +
class Net_batch_normalization(nn.Module):
    def __init__(self):
        super(Net_batch_normalization, self).__init__()
        # Input Block

        
        # Input Block 
    
    # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout2d(dropout_value)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout2d(dropout_value)
        )

        # TRANSITION BLOCK 1
        self.transitionblock3 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(1, 1), padding=0, bias=bias)

        
        self.pool1 = nn.MaxPool2d(2, 2)

        
        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout_value)
        )
        
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout_value)
        )

        # TRANSITION BLOCK 2
        self.transitionblock6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=bias)

        self.pool2 = nn.MaxPool2d(2, 2)

        
        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout_value)
        )
        
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(dropout_value)
        )
        
        
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(dropout_value)
        )

        # OUTPUT BLOCK
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.convblock10 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=bias)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.transitionblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)




class Net_group_norm(nn.Module):
    def __init__(self):
        super(Net_group_norm, self).__init__()
        # Input Block

        
        # Input Block 
    
    # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(16, 48),
            nn.Dropout2d(dropout_value)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(16, 48),
            nn.Dropout2d(dropout_value)
        )

        # TRANSITION BLOCK 1
        self.transitionblock3 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(1, 1), padding=0, bias=bias)

        
        self.pool1 = nn.MaxPool2d(2, 2)

        
        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            nn.Dropout2d(dropout_value)
        )
        
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            nn.Dropout2d(dropout_value)
        )

        # TRANSITION BLOCK 2
        self.transitionblock6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=bias)

        self.pool2 = nn.MaxPool2d(2, 2)

        
        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Dropout2d(dropout_value)
        )
        
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(2, 8),
            nn.Dropout2d(dropout_value)
        )
        
        
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(2, 10),
            nn.Dropout2d(dropout_value)
        )

        # OUTPUT BLOCK
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.convblock10 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=bias)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.transitionblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)



class Net_layer_normalization(nn.Module):
    def __init__(self):
        super(Net_layer_normalization, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([10, 32, 32]),
            nn.Dropout2d(dropout_value)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([8, 32, 32]),
            nn.Dropout2d(dropout_value)
        )

        self.transitionblock3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), padding=0, bias=bias)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([8, 16, 16]),
            nn.Dropout2d(dropout_value)
        )
        
        self.transitionblock6 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), padding=0, bias=bias)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([8, 8, 8]),
            nn.Dropout2d(dropout_value)
        )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([4, 8, 8]),
            nn.Dropout2d(dropout_value)
        )

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([10, 8, 8]),
            nn.Dropout2d(dropout_value)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.convblock10 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=bias)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.transitionblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)




class Net_s6(nn.Module):
    def __init__(self):
        super(Net_s6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128,
        kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        
        self.tns1 = nn.Conv2d(in_channels=128, out_channels=4,
        kernel_size=1, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16,
        kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32,
        kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
       
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.tns2 = nn.Conv2d(in_channels=32, out_channels=16,
        kernel_size=1, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=16)
        
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=32,
        kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=32)
        
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=10,
        kernel_size=1, padding=1)
        self.gpool = nn.AvgPool2d(kernel_size=7)
        self.drop = nn.Dropout2d(0.1)

    
    def forward(self, x):
        x = self.tns1(self.drop(self.bn1(F.relu(self.conv1(x)))))
        x = self.drop(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop(self.bn3(F.relu(self.conv3(x))))
        x = self.drop(self.bn4(F.relu(self.conv4(x))))
        x = self.tns2(self.pool2(x))
        x = self.drop(self.bn5(F.relu(self.conv5(x))))
        x = self.drop(self.bn6(F.relu(self.conv6(x))))
        x = self.conv7(x)
        x = self.gpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net_s7(nn.Module):
    def __init__(self):
        super(Net_s7, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), padding=0, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        
        
        
        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=bias),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        
        
        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        
        
        
        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=bias),
        ) # output_size = 24
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 12        
        
        # CONVOLUTION BLOCK 3                
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 6
              

        # OUTPUT BLOCK
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1,1))) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=bias),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
# -




class Net_group_norm(nn.Module):
    def __init__(self):
        super(Net_group_norm, self).__init__()
        # Input Block

        
        # Input Block 
    
    # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(16, 48),
            nn.Dropout2d(dropout_value)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(16, 48),
            nn.Dropout2d(dropout_value)
        )

        # TRANSITION BLOCK 1
        self.transitionblock3 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(1, 1), padding=0, bias=bias)

        
        self.pool1 = nn.MaxPool2d(2, 2)

        
        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            nn.Dropout2d(dropout_value)
        )
        
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            nn.Dropout2d(dropout_value)
        )

        # TRANSITION BLOCK 2
        self.transitionblock6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=bias)

        self.pool2 = nn.MaxPool2d(2, 2)

        
        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Dropout2d(dropout_value)
        )
        
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(2, 8),
            nn.Dropout2d(dropout_value)
        )
        
        
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.GroupNorm(2, 10),
            nn.Dropout2d(dropout_value)
        )

        # OUTPUT BLOCK
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.convblock10 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=bias)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.transitionblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)



class Net_layer_normalization(nn.Module):
    def __init__(self):
        super(Net_layer_normalization, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([10, 32, 32]),
            nn.Dropout2d(dropout_value)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([8, 32, 32]),
            nn.Dropout2d(dropout_value)
        )

        self.transitionblock3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), padding=0, bias=bias)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([8, 16, 16]),
            nn.Dropout2d(dropout_value)
        )
        
        self.transitionblock6 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), padding=0, bias=bias)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([8, 8, 8]),
            nn.Dropout2d(dropout_value)
        )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([4, 8, 8]),
            nn.Dropout2d(dropout_value)
        )

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=(3, 3), padding=1, bias=bias),
            nn.ReLU(),
            nn.LayerNorm([10, 8, 8]),
            nn.Dropout2d(dropout_value)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.convblock10 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=bias)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.transitionblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)




class Net_s6(nn.Module):
    def __init__(self):
        super(Net_s6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128,
        kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        
        self.tns1 = nn.Conv2d(in_channels=128, out_channels=4,
        kernel_size=1, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16,
        kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32,
        kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
       
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.tns2 = nn.Conv2d(in_channels=32, out_channels=16,
        kernel_size=1, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=16)
        
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=32,
        kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=32)
        
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=10,
        kernel_size=1, padding=1)
        self.gpool = nn.AvgPool2d(kernel_size=7)
        self.drop = nn.Dropout2d(0.1)

    
    def forward(self, x):
        x = self.tns1(self.drop(self.bn1(F.relu(self.conv1(x)))))
        x = self.drop(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop(self.bn3(F.relu(self.conv3(x))))
        x = self.drop(self.bn4(F.relu(self.conv4(x))))
        x = self.tns2(self.pool2(x))
        x = self.drop(self.bn5(F.relu(self.conv5(x))))
        x = self.drop(self.bn6(F.relu(self.conv6(x))))
        x = self.conv7(x)
        x = self.gpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)

