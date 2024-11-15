import torch.utils.data

from dataset import HairDataset
from transformer import TransformerEncoder
from loss import CalculateLoss
import cv2
csv_path = './data/hair_class.csv'
BATCH_SIZE = 4
EPOCH_COUNT = 50
LR = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transform(image, label):
    image = cv2.resize(image, (224, 224))
    image = image/255
    image = torch.tensor(image).permute(2, 0, 1).float()
    label = torch.tensor(label)
    return image, label

if __name__ == '__main__':

    model = TransformerEncoder()
    criterion = CalculateLoss()
    model = model.to(device)
    criterion.to(device)


    hair_dataset = HairDataset(csv_path, transform=transform)
    hair_dataloader = torch.utils.data.DataLoader(hair_dataset,BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
    print('Finished Training')
    torch.save(model.state_dict(), './checkpoints/vit.pth')


