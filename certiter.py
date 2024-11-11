
import torch.utils.data
import numpy as np
from dataset import HairDataset
from backbone import VGG19
from loss import CalculateLoss
import cv2

csv_path = './data/hair_class.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def transform(image, label):
    image = cv2.resize(image, (224, 224))
    image = image/255
    image = torch.tensor(image).permute(2, 0, 1).float()
    label = torch.tensor(label)
    return image, label

def bitmap_convert_label(bitmap):
    return np.argmax(bitmap)

if __name__ == '__main__':

    model = VGG19(n_class=4)
    model_dict = model.state_dict()
    model_dict_w = torch.load('./checkpoints/vgg19.pth')
    for k, v in model_dict.items():
        if k in model_dict_w:
            model_dict[k].copy_(model_dict_w[k])

    criterion = CalculateLoss()
    model.to(device)
    model.eval()
    criterion.to(device)

    hair_dataset = HairDataset(csv_path, transform=transform)
    hair_dataloader = torch.utils.data.DataLoader(hair_dataset,1, shuffle=True)
    count = 0
    error_count = 0
    for image_path, image, label in hair_dataloader:
        image = image.to(device)
        output = model(image)
        output = output.cpu().detach().numpy()
        predict = bitmap_convert_label(output)
        label = bitmap_convert_label(label.cpu().detach().numpy())
        count = count + 1

        if predict != label:
            error_count = error_count + 1

    print('predict:{}/{} = {}'.format(count-error_count, count, 1 -  error_count/ count))


