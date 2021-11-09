import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

## this is the file for traning the model


import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_Simple

# from network import C3D_model, R2Plus1D_model, R3D_model, C3D_v2_model, C3D_BN_model, C3D_LSTM

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 50  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 10  # Store a model every snapshot epochs
lr = 1e-4  # Learning rate
weight_decay = 1e-4

dataset = 'ucf101'  # Options: hmdb51 or ucf101

time_step = 2
frames = 8
totalFrames = frames * time_step

if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]


if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
# save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(10))

modelName = 'ConvLstm'  # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'ConvLstm':  # this model is ours
        # this model contains 3dcnn - gated 3dcnn - 2layers of 3dcnn - 1 gated 3dcnn - 4 dense layers
        # maxpooling id done withing 3dcnn layers and average pooling between 3dcnn and gated 3dcnn
        model = C3D_Simple.C3D_AttNAtt2(sample_size=112, num_classes=101,
                                        lstm_hidden_size=512, lstm_num_layers=1)

        train_params = model.parameters()

    elif modelName == 'Conv_simple':  # this model is ours
        # this model contains 3dcnn - gated 3dcnn - 4 dense layers
        model = C3D_Simple.C3D_LSTM_Simple(sample_size=112, num_classes=101,
                                       lstm_hidden_size=512, lstm_num_layers=1)
        train_params = model.parameters()

    elif modelName == 'Conv_att3':  # this model is ours
        # this layers contains 3layers of 3dcnn - 1 gated 3dcnn and 4 dense layers
        model = C3D_Simple.C3D_LSTM(sample_size=112, num_classes=101,
                                    lstm_hidden_size=512, lstm_num_layers=1)
        train_params = model.parameters()

    elif modelName == 'Conv_attnatt':  # this model is ours
        # this model contains 3dcnn - gated 3dcnn - 2layers of 3dcnn - 1 gated 3dcnn - 4 dense layers
        model = C3D_Simple.C3D_AttNAtt(sample_size=112, num_classes=101,
                                       lstm_hidden_size=512, lstm_num_layers=1)
        train_params = model.parameters()

    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4,
                                          gamma=0.95)

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        # model.to(device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=totalFrames), batch_size=32,
                                  prefetch_factor=32, pin_memory=True, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=totalFrames), batch_size=32,
                                prefetch_factor=32, pin_memory=True, num_workers=8)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=totalFrames), batch_size=32,
                                 prefetch_factor=32, pin_memory=True, num_workers=8)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'train':

                scheduler.step()
                model.train()
            else:
                model.eval()

            for t, (inputs, labels) in enumerate(tqdm(trainval_loaders[phase])):

                inputs = Variable(inputs, requires_grad=True).to(device)

                labels = Variable(labels).to(device)

                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)


                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                labels = labels.long()
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                '''if inputs % 100 == 0:
                    print('Train Step: {} ({:.4f}%)\tLoss: {:.4f}'.format(
                        num_epochs, running_corrects,running_loss
                    ))'''

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()