import os
import torch

output_dir = 'output'

def save_checkpoint(states, is_best, filename='checkpoint.pth'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # torch.save(states, os.path.join(output_dir, filename))
    torch.save(states, os.path.join(output_dir, 'latest.pth'))

    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))


def resume_checkpoint(model,optimizer):
    model_state_file = os.path.join(output_dir, 'latest.pth')
    last_epoch, best_acc = 0,0
    if os.path.exists(model_state_file):
        checkpoint = torch.load(model_state_file)

        last_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint Epoch:[%d] | Acc:%.3f%% | Best Acc:%.3f%%"%(checkpoint['epoch'],checkpoint['acc']*100,checkpoint['best_acc']*100))
    else:
        print("=> no checkpoint found")

    return model,optimizer,last_epoch,best_acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count