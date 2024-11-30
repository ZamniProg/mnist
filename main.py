import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

trans = tv.transforms.Compose([
    tv.transforms.ToTensor()
])

train_ds = tv.datasets.MNIST("./datasets", train=True, download=True, transform=trans)
test_ds = tv.datasets.MNIST("./datasets", train=False, download=True, transform=trans)

batch_size = 16

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.flat(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out


# Модель
model = NeuralNetwork()


def train(dl, epochs):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        average_loss = 0
        average_acc = 0
        for img, label in dl:
            optimizer.zero_grad()

            pred = model(img)

            acc = accuracy(pred, label)
            average_acc += acc

            loss = loss_fn(pred, label)
            average_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch + 1}, Loss: {average_loss / len(dl):.4f}, Accuracy: {average_acc / len(dl):.4f}")


def accuracy(pred, label):
    pred_labels = pred.argmax(dim=1)
    return (pred_labels == label).float().mean().item()


def evaluate(dl):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in dl:
            pred = model(img)
            pred_labels = pred.argmax(dim=1)
            correct += (pred_labels == label).sum().item()
            total += label.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")


def predict(image):
    model.eval()

    with torch.no_grad():
        if isinstance(image, torch.Tensor):
            img_tensor = image.unsqueeze(0)
        else:
            img_tensor = trans(image).unsqueeze(0)

        output = model(img_tensor)
        pred_label = output.argmax(dim=1).item()

    return pred_label


def show_image(image, label):
    image = image.squeeze(0)
    image = image.numpy()

    if image.ndim == 1:
        image = image.reshape(28, 28)

    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted label: {label}")
    plt.axis('off')
    plt.show()

train(train_loader, 10)
evaluate(test_loader)

example_image, _ = test_ds[0]

predicted_label = predict(example_image)

print(f"Predicted label for the example image: {predicted_label}")

show_image(example_image, predicted_label)
