from torch import nn

class CalculateLoss(nn.Module):
    def __init__(self):
        super(CalculateLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        ce_loss = self.criterion(output, target)
        return ce_loss
