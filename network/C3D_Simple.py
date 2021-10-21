import torch
import torch.nn as nn



# from mypath import path
import torch.nn.functional as F


class C3D_LSTM(nn.Module):
    """
    The C3D network.
    """


    def __init__(self, sample_size, num_classes,
                 lstm_hidden_size, lstm_num_layers):
        ## 2 layers of conv3d and then gated conv3d
        ## 4 layers of linear

        super(C3D_LSTM, self).__init__()

        self.sample_size = sample_size
        self.num_classes = num_classes


        self.conv1=nn.Conv3d(3, 64, kernel_size=(2, 6, 6),stride=(2,2,2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(2, 3, 3), stride=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(2, 3, 3),stride=(2,2,2))
        self.att_at3 = nn.Conv3d(128, 256, kernel_size=(2, 3, 3),stride=(2,2,2))  

        self.pool1=nn.AvgPool3d((2,3,3),(2,2,2))
        self.fc1=nn.Linear(6400,4096)
        self.fc2=nn.Linear(4096,2048)
        self.fc3=nn.Linear(2048,1024)
        self.fc4=nn.Linear(1024,num_classes)

        self.drop1=nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.25)
        self.drop3 = nn.Dropout(0.25)

    def forward(self, x):
        #print(x.shape)

        output=F.relu(self.conv1(x))
        output=F.relu(self.conv2(output))

        output1=F.relu(self.conv3(output))
        gate= F.sigmoid(self.att_at3(output))

        output=torch.mul(output1,gate)
        #print(output.shape) 
        output=self.pool1(output)
        #print("pool1:",output.shape)
        output=self.drop1(F.relu(self.fc1(output.view(x.shape[0],-1))))
        output=self.drop2(F.relu(self.fc2(output)))
        output = self.drop3(F.relu(self.fc3(output)))
        output = self.fc4(output)
        return output



class C3D_AttNAtt(nn.Module): 
    """
    The C3D network.
    """

    def __init__(self, sample_size, num_classes,
                 lstm_hidden_size, lstm_num_layers):
       
        super(C3D_AttNAtt, self).__init__()

        self.sample_size = sample_size
        self.num_classes = num_classes

      

        self.conv1=nn.Conv3d(3, 64, kernel_size=(3, 5, 5),stride=(2,3,3))
        self.att_at1=nn.Conv3d(3, 64, kernel_size=(3, 5,5),stride=(2,3,3))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(2, 3, 3), stride=(1, 2, 2))
        self.conv3=  nn.Conv3d(64, 128, kernel_size=(2, 3, 3), stride=(1, 2, 2))

        self.conv4 = nn.Conv3d(256, 512, kernel_size=(2, 3, 3),stride=(1,2,2))
        self.att_at4 = nn.Conv3d(256, 512, kernel_size=(2, 3, 3),stride=(1,2,2)) 

        self.pool1=nn.AvgPool3d((2,3,3),(2,2,2))
        self.fc1=nn.Linear(9216,4096)
        self.fc2=nn.Linear(4096,2048)
        self.fc3=nn.Linear(2048,512)
        self.fc4=nn.Linear(512,num_classes)

        self.batch1=nn.LayerNorm((128, 6, 17, 17))
        self.batch2 = nn.LayerNorm((128, 6, 17, 17))

        self.norm1=nn.LayerNorm(9216)
        self.norm2 = nn.LayerNorm(4096)
        self.norm3 = nn.LayerNorm(2048)
        self.norm4 = nn.LayerNorm(512)

        self.drop0=nn.Dropout(0.25)
        self.drop1=nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.25)
        self.drop3 = nn.Dropout(0.25)

    def print_shape(self,x,name,debug=False):
        if debug==True:
            print(f'At {name}:',x.shape)

    def forward(self, x,debug=False):
        #print(x.shape)

        output1 = F.relu(self.conv1(x))
        gate = F.sigmoid(self.att_at1(x))
        output = torch.mul(output1, gate)
        self.print_shape(output1,'gate1',debug)

        output1=self.batch1(F.relu(self.conv2(output)))
        self.print_shape(output1, 'conv2', debug)
        output2=self.batch1(F.relu(self.conv3(output)))
        self.print_shape(output2, 'conv3', debug)

        output=torch.cat((output1,output2),axis=1)
        self.print_shape(output, 'conv3', debug)

        output1=F.relu(self.conv4(output))
        gate2= F.sigmoid(self.att_at4(output))

        output=torch.mul(output1,gate2)
        self.print_shape(output, 'gate2', debug)

        output=self.drop0(self.pool1(output).view(x.shape[0],-1))
        self.print_shape(output, 'pool1', debug)

        output=self.norm2(self.drop1(F.relu(self.fc1(self.norm1(output)))))
        output=self.norm3(self.drop2(F.relu(self.fc2(output))))
        output = self.norm4(self.drop3(F.relu(self.fc3(output))))
        output = self.fc4(output)
        return output




class C3D_AttNAtt2(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, sample_size, num_classes,
                 lstm_hidden_size, lstm_num_layers):
      
        super(C3D_AttNAtt2, self).__init__()

        self.sample_size = sample_size
        self.num_classes = num_classes

      

        self.conv1=nn.Conv3d(3, 64, kernel_size=(3, 5, 5),stride=(2,3,3),padding=(1,1,1))
        self.att_at1=nn.Conv3d(3, 64, kernel_size=(3, 5,5),stride=(2,3,3),padding=(1,1,1))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(2, 3, 3), stride=(1, 2, 2),padding=(1,1,1))
        self.conv3=  nn.Conv3d(64, 128, kernel_size=(2, 3, 3), stride=(1, 2, 2),padding=(1,1,1))

        self.conv4 = nn.Conv3d(256, 512, kernel_size=(2, 3, 3),stride=(1,2,2),padding=(1,1,1))
        self.att_at4 = nn.Conv3d(256, 512, kernel_size=(2, 3, 3),stride=(1,2,2),padding=(1,1,1))  #

        self.interPool1=nn.MaxPool3d((2,2,2),(1,2,2)) ##
        self.interPool2=nn.MaxPool3d((2,2,2),(1,2,2))
        self.pool1=nn.AvgPool3d((2,3,3),(2,2,2))

        self.fc1=nn.Linear(8192,4096)
        self.fc2=nn.Linear(4096,2048)
        self.fc3=nn.Linear(2048,512)
        self.fc4=nn.Linear(512,num_classes)

        self.cnn_norm1=nn.LayerNorm((128, 9, 19, 19))
        self.cnn_norm2 = nn.LayerNorm((128,9, 19, 19))

        self.norm1=nn.LayerNorm(8192)
        self.norm2 = nn.LayerNorm(4096)
        self.norm3 = nn.LayerNorm(2048)
        self.norm4 = nn.LayerNorm(512)

        self.drop0=nn.Dropout(0.25)
        self.drop1=nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.25)
        self.drop3 = nn.Dropout(0.25)

    def print_shape(self,x,name,debug=False):
        if debug==True:
            print(f'At {name}:',x.shape)

    def forward(self, x,debug=False):
        #print(x.shape)

        output1 = F.relu(self.conv1(x))
        gate = F.sigmoid(self.att_at1(x))
        output = torch.mul(output1, gate)
        self.print_shape(output1,'gate1',debug)

        output1=F.relu(self.conv2(output))
        self.print_shape(output1, 'conv2', debug)
        output1=self.interPool1(self.cnn_norm1(output1))
        self.print_shape(output1, 'conv2_interPool1', debug)

        output2=F.relu(self.conv3(output))
        self.print_shape(output2, 'conv3', debug)
        output2=self.interPool1(self.cnn_norm2(output2))
        self.print_shape(output2, 'conv3_interPool2', debug)

        output=torch.cat((output1,output2),axis=1)
        self.print_shape(output, 'conv3', debug)

        output1=F.relu(self.conv4(output))
        gate2= F.sigmoid(self.att_at4(output))

        output=torch.mul(output1,gate2)
        self.print_shape(output, 'gate2', debug)

        output=self.drop0(self.pool1(output).view(x.shape[0],-1))
        self.print_shape(output, 'pool1', debug)

        output=self.norm2(self.drop1(F.relu(self.fc1(self.norm1(output)))))
        output=self.norm3(self.drop2(F.relu(self.fc2(output))))
        output = self.norm4(self.drop3(F.relu(self.fc3(output))))
        output = self.fc4(output)
        return output


## this model made 82% on val set and 82% on test set by epoch 20 , 
##  first layer gated Cnn3D , 2-3 layers just CNN3D and last layer gated Cnn3D, plus Layer Normalization on CNN3D (2-3 layers) and layer normalization on Linear Layers (all)
## plus Dropoouts

if __name__ == "__main__":
    time_step = 2  #
    frames = 8
    inputs = torch.rand(5, 3, frames * time_step, 112, 112)


    net = C3D_AttNAtt2(sample_size=112, num_classes=101,
                   lstm_hidden_size=512, lstm_num_layers=3)
    net = net.cuda()

    print(net)

    outputs = net.forward(inputs.cuda(),True)
    print(outputs.size())


