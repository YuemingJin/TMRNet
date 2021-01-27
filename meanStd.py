import numpy as np
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision import transforms
import pickle
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class CholecDataset(Dataset):
    def __init__(self, file_paths, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs

    def __len__(self):
        return len(self.file_paths)


with open('./train_val_paths_labels.pkl', 'rb') as f:
    test_paths_labels = pickle.load(f)

test_paths = test_paths_labels[1]
print(test_paths[0])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
])
test_dataset = CholecDataset(test_paths, test_transforms)


print('test_paths   : {:6d}'.format(len(test_paths)))

pop_mean = []
pop_std0 = []
# print(dataset)
for i in range(len(test_dataset)):
    # print(img, label)
    # shape (batch_size, 3, height, width)
    print(i)
    img = test_dataset[i]
    numpy_image = img.numpy()

    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(1, 2))
    batch_std0 = np.std(numpy_image, axis=(1, 2))

    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std0 = np.array(pop_std0).mean(axis=0)

print(pop_mean, pop_std0)
