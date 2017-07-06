import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import LBFGS
import torchvision
from torchvision import transforms
from torch.backends import cudnn
cudnn.benchmark = True

style_img = 'wave.png'
content_img = 'brad_pitt.png'

def gram(input):
    b,c,h,w = input.size()
    F = input.view(b, c, h*w)
    G = torch.bmm(F, F.transpose(1,2))
    G.div_(h*w)
    return G

style_layers = [2, 7, 12, 19]
content_layers = [19]

style_weights = [1e0 for n in [64,128,256,512]]
content_weights = [1e0]

def vgg(inputs, model):
    '''VGG definition with style and content outputs.
    '''
    style, content = [], []

    def block(x, ids):
        for i in ids:
            x = F.relu(F.conv2d(x, Variable(model.features[i].weight.data.cuda()),Variable(model.features[i].bias.data.cuda()), 1, 1), inplace=True)
            if i in style_layers:
                style.append(gram(x))
            if i in content_layers:
                content.append(x)
        return F.max_pool2d(x, 2, 2)

    o = block(inputs, [0, 2])
    o = block(o, [5, 7])
    o = block(o, [10, 12, 14])
    o = block(o, [17, 19, 21])
    o = block(o, [24, 26, 28])
    return style, content

img_size = 512
tr_mean, tr_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
prep = transforms.Compose([transforms.Scale(img_size),
                           transforms.ToTensor(),
                           transforms.Normalize(tr_mean, tr_std),
                          ])

def postp(tensor): # to clip results in the range [0,1]
    mu = torch.Tensor(tr_mean).view(-1,1,1).expand_as(tensor)
    sigma = torch.Tensor(tr_std).view(-1,1,1).expand_as(tensor)
    img = (tensor * sigma + mu).clamp(0, 1)
    return img

#load pretrained vgg16 model
model = torchvision.models.vgg16(pretrained=True)

def load_img(path):
    return Image.open(path)

imgs = [load_img(style_img), load_img(content_img)]
imgs_torch = [Variable(prep(img).unsqueeze(0).cuda()) for img in imgs]
style_image, content_image = imgs_torch

style_targets = vgg(style_image, model)[0]
content_targets = vgg(content_image, model)[1]

#run style transfer
max_iter = 500
show_iter = 50
opt_img = Variable(content_image.data.clone(), requires_grad=True)
optimizer = LBFGS([opt_img]);
n_iter=[0]

def l1_loss(x, y):
    return torch.abs(x - y).mean()

while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        style, content = vgg(opt_img, model)
        style_loss = sum(alpha * l1_loss(u, v)
                         for alpha, u, v in zip(style_weights, style, style_targets))
        content_loss = sum(beta * l1_loss(u, v)
                           for beta, u, v in zip(content_weights, content, content_targets))
        loss = style_loss + content_loss
        loss.backward()
        n_iter[0]+=1
        if n_iter[0]%show_iter == (show_iter-1):
            print('Iteration: %d, style loss: %f, content loss: %f'%(n_iter[0]+1,style_loss.data[0], content_loss.data[0]))
            out_img = postp(opt_img.data[0].cpu().squeeze())
            torchvision.utils.save_image(out_img,'out_%d.png'%(n_iter[0]+1))

        return loss
    
    optimizer.step(closure)
