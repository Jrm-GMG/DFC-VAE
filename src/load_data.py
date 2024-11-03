import torchvision
import torch
from torchvision import transforms as T

def load_data(batch_size=50):
    trans_comp = T.Compose([T.Resize([128,128]),
                            T.ToTensor()])
    
    data_train = torchvision.datasets.CelebA("./celebA/",split='train',download=True,transform=trans_comp,target_type='attr')
    data_test = torchvision.datasets.CelebA("./celebA/",split='test',download=True,transform=trans_comp,target_type='attr')
    data_validation = torchvision.datasets.CelebA("./celebA/",split='valid',download=True,transform=trans_comp,target_type='attr')

    train_dataset = torch.utils.data.DataLoader(data_train,
                                                batch_size=batch_size,
                                                shuffle=True,)

    test_dataset = torch.utils.data.DataLoader(data_test,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )
    
    validation_dataset = torch.utils.data.DataLoader(data_validation,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )
    return train_dataset, validation_dataset, test_dataset