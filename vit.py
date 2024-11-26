import numpy as np
import torch.utils.data
from dataset import HairDataset
from transformer import TransformerEncoder
from loss import CalculateLoss
import cv2
from visdom import Visdom

csv_path = './data/hair_class.csv'
BATCH_SIZE = 128
EPOCH_COUNT = 50
LR = 5e-1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transform(image, label):
    image = cv2.resize(image, (224, 224))
    image = image/255
    image = torch.tensor(image).permute(2, 0, 1).float()
    label = torch.tensor(label)
    return image, label

if __name__ == '__main__':

    model = TransformerEncoder(layer_count=12)
    # model_dict = model.state_dict()
    # model_dict_w = torch.load('checkpoints/vit-epoch-50_5lr2-sgd.pth')
    # for k, v in model_dict.items():
    #     if k in model_dict_w:
    #         model_dict[k].copy_(model_dict_w[k])

    criterion = CalculateLoss()
    model = model.to(device)
    criterion.to(device)
    model.train()
    print(model)
    # vis = Visdom()
    hair_dataset = HairDataset(csv_path, transform=transform)
    hair_dataloader = torch.utils.data.DataLoader(hair_dataset,BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(EPOCH_COUNT):
        loss_total = 0
        counter = 0
        for image_path, image, label in hair_dataloader:
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion(output, label)
            print('Epoch:{} Current loss: {:.6f}'.format(epoch ,loss.item()))
            loss_total += loss.item()
            counter = counter + 1
            loss.backward()
            optimizer.step()
        print('Epoch %d, Loss: %f' % (epoch, loss_total/counter))
        # vis.line(X=np.array([epoch]), Y=np.array([loss_total/counter]), win='epoch loss', update='append')
    print('Finished Training')
    torch.save(model.state_dict(), 'checkpoints/vit-epoch-50_adam_5lr4-sgd.pth')


