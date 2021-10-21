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
        super(C3D_LSTM, self).__init__()

        self.sample_size = sample_size
        self.num_classes = num_classes


        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 6, 6),stride=(2,3,3))
        self.att = nn.Conv3d(3, 64, kernel_size=(3, 6, 6),stride=(2,3,3))

        self.pool1=nn.AvgPool3d((2,6,6),(2,3,3))
        self.fc1=nn.Linear(23232,4096*2)
        self.fc2=nn.Linear(4096*2,2048)
        self.fc3=nn.Linear(2048,1024)
        self.fc4=nn.Linear(1024,num_classes)

        self.drop1=nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.25)
        self.drop3 = nn.Dropout(0.25)

    def forward(self, x):
        #print(x.shape)

        output=F.relu(self.conv1(x))
        gate= F.sigmoid(self.att(x))

        output=torch.mul(output,gate)
        #print(output.shape)
        output=self.pool1(output)
        #print("pool1:",output.shape)
        output=self.drop1(F.relu(self.fc1(output.view(x.shape[0],-1))))
        output=self.drop2(F.relu(self.fc2(output)))
        output = self.drop3(F.relu(self.fc3(output)))
        output = self.fc4(output)
        return output


if __name__ == "__main__":
    time_step = 2  #
    frames = 8
    inputs = torch.rand(5, 3, frames * time_step, 112, 112)
    

    net = C3D_LSTM(sample_size=112, num_classes=101,
                   lstm_hidden_size=512, lstm_num_layers=3)
    net = net.cuda()

    print(net)

    outputs = net.forward(inputs.cuda())
    print(outputs.size())


