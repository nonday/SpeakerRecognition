import torch.nn as nn
import torch
import torch.optim as optim

from model import ResCNN
from utils import save_checkpoint, resume_checkpoint, AverageMeter

from dataset import Dataset


def train(model, optimizer, criterion, data_loader, epoch, epochs,print_freq=50):
    #
    model.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        targets = targets.view(-1)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward and loss
        out = model(inputs)
        loss = criterion(out, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # information
        train_loss.update(loss.item(), n=targets.size(0))

        _, predicted = out.max(1)
        batch_correct = predicted.eq(targets).sum().item()
        batch_total = targets.size(0)
        batch_acc = batch_correct / batch_total

        train_acc.update(batch_acc, n=targets.size(0))

        if batch_idx % print_freq == 0:
            print('Epoch:[%d/%d] Batch:[%d/%d]'%(epoch,epochs,batch_idx+1,len(data_loader)),
                  'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                  train_loss.avg, 100. * train_acc.avg, train_acc.sum, train_acc.count))

    return train_loss.avg, train_acc






if __name__ == '__main__':

    train_data_loader = torch.utils.data.DataLoader(dataset = Dataset(path = 'aishell/wav1-20', nframes = 160),
                                                    batch_size=32,
                                                    shuffle=True,
                                                    num_workers=2)

    # build model
    n_speakers = 20
    model = ResCNN(inp=3, num_classes=n_speakers)
    if torch.cuda.is_available():
        model = model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # loss function
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # resume model from last epoch
    last_epoch = 0
    best_acc = 0
    RESUME = True
    if RESUME:
        model, optimizer, last_epoch, best_acc = resume_checkpoint(model, optimizer)

    # Epoch iter
    epochs = 50
    for epoch in range(last_epoch + 1, epochs + 1):
        # train
        avg_loss,train_acc = train(model, optimizer, criterion, train_data_loader, epoch, epochs, print_freq=10)

        is_best = best_acc < train_acc.avg
        if is_best:
            best_acc = train_acc.avg

        print('Epoch:[%d/%d] || Train Loss(Avg): %.3f | Acc: %.3f%% (%d/%d) | Best:' % (
              epoch, epochs, avg_loss,
              100. * train_acc.avg, train_acc.sum, train_acc.count),
              is_best)

        save_checkpoint(
            {"state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "epoch": epoch,
             "best_acc": best_acc,
             "acc":train_acc.avg},
             is_best,
             'checkpoint_%d.pth' % (epoch)
        )


