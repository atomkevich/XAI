import torch
import torch.nn as nn
import loader
from PIL import ImageFile
import cv2
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import model
import numpy as np
import glob


def train_added_layers(model, loader, num_epochs):
    trainable_parameters = []
    for name, p in model.named_parameters():
        if "fc" in name:
            trainable_parameters.append(p)
    optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.1, momentum=1e-5)
    criterion = nn.CrossEntropyLoss()
    total_step = len(loader)
    loss_list = []
    acc_list = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))
        cam = np.matmul(weight[idx], beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


if __name__ == '__main__':
    class_num = 2
    batch_size = 8
    num_epochs = 1
    IMG_URL = "/Users/alinatamkevich/images/cats"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_loader = loader.generate_loader(batch_size)
    mod, model = model.build_model(class_num)
    train_added_layers(model, train_loader, num_epochs)
    params = list(model.parameters())
    weight = np.squeeze(params[-2].data.numpy())

    predicted_labels = []
    for fname in glob.glob(IMG_URL + "/*.jpeg"):
        img_pil = Image.open(fname)
        img_tensor = preprocess(img_pil)
        img_variable = Variable(img_tensor.unsqueeze(0))
        logit = model(img_variable)

        h_x = logit.data.squeeze()

        probs, idx = h_x.sort(0, True)
        probs = probs.detach().numpy()
        idx = idx.numpy()

        predicted_labels.append(idx[0])
        predicted = train_loader.dataset.classes[idx[0]]

        print("Target: " + fname + " | Predicted: " + predicted)

        features_blobs = mod(img_variable)
        features_blobs1 = features_blobs.cpu().detach().numpy()
        CAMs = return_CAM(features_blobs1, weight, [idx[0]])

        img = cv2.imread(fname)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + img * 0.5

        cv2.imwrite("image_1.jpg", result)