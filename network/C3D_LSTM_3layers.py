import torch
import torch.nn as nn

## this is the model file for 3d cnn and lstm combination
## 3dCNN + lstm - like 3d cnn will extract feartures from 16 frames (we can choose, generally its 16 frames in 3d cnn model) n then pass the features to the lstm after getting the features from few chuncks (like 2-3 or 3-6s 16 frames chunks) of 16 frames n then predict the activtiy class


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
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(2, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(2, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(2, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(2, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(2, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(2, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 4, 4), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1024)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=False,
            dropout=0.2,
        )

        self.relu = nn.ReLU()
        self.fc9 = nn.Linear(self.lstm_hidden_size, self.num_classes)

    def forward(self, x):
        # passing CNN parameters
        cnn_seq = []
        # print('Shape of Input Tensor:', x.shape)

        for t in range(x.shape[0]):
            out = self.relu(self.conv1(x[t]))
            out = self.pool1(out)

            out = self.relu(self.conv2(out))
            out = self.pool2(out)

            out = self.relu(self.conv3a(out))
            out = self.relu(self.conv3b(out))
            out = self.pool3(out)

            out = self.relu(self.conv4a(out))
            out = self.relu(self.conv4b(out))
            out = self.pool4(out)

            out = self.relu(self.conv5a(out))
            out = self.relu(self.conv5b(out))
            out = self.pool5(out)

            out = out.view(x.shape[1], -1)
            out = self.relu(self.fc6(out))
            out = self.dropout(out)
            out = self.relu(self.fc7(out))
            out = self.dropout2(out)
            out = self.relu(self.fc8(out))
            out = self.dropout3(out)

            # print('Output shape after Conv', out.shape)
            # print("time step for lstm", t)
            cnn_seq.append(out)
        cnn_seq = torch.stack(cnn_seq, 0)

        # LSTMss
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_seq, None)

        out = self.fc9(out[-1, :, :])

        return out


if __name__ == "__main__":
    time_step = 2  #
    frames = 8
    inputs = torch.rand(1, 3, frames * time_step, 112, 112)
    # TotalFrame=frames*time_steps
    # old shape (batch, channel,TotalFrames,w,h) -> new shape (time_step,batch,channel,frames,w,h)
    inputs = torch.stack(torch.split(inputs, frames, dim=2))

    net = C3D_LSTM(sample_size=112, num_classes=101,
                   lstm_hidden_size=512, lstm_num_layers=3)
    net = net.cuda()

    print(net)

    outputs = net.forward(inputs.cuda())
    print(outputs.size())


