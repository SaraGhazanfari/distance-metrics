import torch
import time
import torchvision
from torchvision import transforms as T

from torchvision import models as torchvision_models
import torch.nn as nn

# basic parameters
cuda_device = torch.device('cuda:0')


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.tensor([-.030, -.088, -.188], device=cuda_device)[None, :, None, None])
        self.register_buffer('scale', torch.tensor([.458, .448, .450], device=cuda_device)[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=True):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers).cuda()

    def forward(self, x):
        return self.model(x)


class ImageNetNormalizer(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        mean = torch.tensor(self.mean, device=x.device)
        std = torch.tensor(self.std, device=x.device)

        return (
                (x - mean[None, :, None, None]) /
                std[None, :, None, None]
        )


class AlexNetFeatureModel(nn.Module):
    model: torchvision_models.AlexNet

    def __init__(self):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.normalizer = ImageNetNormalizer()
        self.model = torchvision_models.alexnet(pretrained=True).cuda().eval()

        assert len(self.model.features) == 13
        self.layer1 = nn.Sequential(self.model.features[:2])
        self.layer2 = nn.Sequential(self.model.features[2:5])
        self.layer3 = nn.Sequential(self.model.features[5:8])
        self.layer4 = nn.Sequential(self.model.features[8:10])
        self.layer5 = nn.Sequential(self.model.features[10:12])

        self.chns = [64, 192, 384, 256, 256]
        self.lin0 = NetLinLayer(self.chns[0])
        self.lin1 = NetLinLayer(self.chns[1])
        self.lin2 = NetLinLayer(self.chns[2])
        self.lin3 = NetLinLayer(self.chns[3])
        self.lin4 = NetLinLayer(self.chns[4])

        self.only_conv_layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        [layer.eval() for layer in self.only_conv_layers]
        self.load_state_dict(torch.load('../checkpoints/alex.pth', map_location='cuda'), strict=False)

    def normalize_tensor(self, in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps).detach()

    def forward(self, x):
        # x = self.normalizer(x)
        x = self.scaling_layer(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x_layer5 = self.layer5(x_layer4)

        return (self.lin0(self.normalize_tensor(x_layer1)), self.lin1(self.normalize_tensor(x_layer2)),
                self.lin2(self.normalize_tensor(x_layer3)), self.lin3(self.normalize_tensor(x_layer4)),
                self.lin4(self.normalize_tensor(x_layer5)))


class LPIPS_Metric(nn.Module):
    def __init__(self):
        super().__init__()

    def spatial_average(self, arr):
        return arr.mean([2, 3], keepdim=True)

    def forward(self, img1, img2):
        for img_parts in img1:
            img_parts.detach()
        for img_parts in img2:
            img_parts.detach()
        img_1_and_img_2_diff, res = {}, {}
        for idx in range(5):
            img_1_and_img_2_diff[idx] = (img1[idx] - img2[idx]) ** 2
            img_1_and_img_2_diff[idx].detach()

            res[idx] = self.spatial_average(img_1_and_img_2_diff[idx]).reshape(-1).detach()

            # img_1_and_img_2_diff[idx] = (img1[idx]-img2[idx])**2
            # img_1_and_img_2_diff[idx].detach()

            # res[idx] = self.spatial_average(self.only_conv_layers[idx](
            #     img_1_and_img_2_diff[idx])).reshape(-1).detach()

        res_sum = 0
        for i in range(len(res)):
            res_sum += res[i]
        return res_sum.detach()


transform = T.Compose(
    [
        T.ToTensor()
    ]
)

train_dataset = torchvision.datasets.SVHN(root='./', split='train', transform=transform, download=True)
sorted_dataset = sorted(train_dataset, key=lambda x: x[1])
sorted_dataset = torch.stack([data for data, _ in sorted_dataset])

# Calculating distance between points in an efficient way.
start_time = time.time()
lpips_distance_matrix = torch.zeros((sorted_dataset.shape[0], sorted_dataset.shape[0]))
alex_net = AlexNetFeatureModel().eval()
lpips_metric = LPIPS_Metric().eval()

for idx, data in enumerate(sorted_dataset):
    lpips_distance_matrix[idx, idx:] = lpips_metric(alex_net(data.cuda()), alex_net(sorted_dataset[idx:].cuda())).cpu()

    if idx % 100 == 99:
        import sys

        with open('file', 'w') as sys.stdout:
            print('Data index: {idx}, time: {time}'.format(idx=idx + 1, time=(time.time() - start_time) / 60))
torch.save(lpips_distance_matrix, 'svhn-lpips-matrix.pt')
