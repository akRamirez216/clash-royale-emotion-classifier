"""
This module defines the ResNet18 architecture for emotion detection using PyTorch.
It includes the model definition, loading of trained weights, and a prediction function.

Authors: Asher Kelly, Zai Yang
Date: December 13, 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Must match your training class order:
CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Basic Block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Docstring for __init__
        
        :param self: self
        :param in_channels: specifies the number of input channels.
        :param out_channels: specifies the number of output channels.
        :param stride: specifies the stride for the convolutional layer. Default is 1.
        :return: None
        """
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    # Forward pass
    def forward(self, x):
        """
        Docstring for forward
        
        :param self: self
        :param x: input tensor
        :return: output tensor after applying the block operations
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet Architecture
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        """
        Docstring for __init__
        
        :param self: self
        :param block: Specifies block type to be used in the ResNet architecture
        :param num_blocks: Specifies the number of blocks in each layer of the ResNet.
        :param num_classes: 
        """
        super(ResNet, self).__init__()
        self.in_channels = 32

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            1, 32,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)

        # ResNet layers
        self.layer1 = self._make_layer(block, 32,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    # Create ResNet layers
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Docstring for _make_layer
        
        :param self: self
        :param block: block type to be used in the layer
        :param out_channels: number of output channels for the layer
        :param num_blocks: number of blocks in the layer
        :param stride: stride for the first block in the layer
        :return: A sequential container of the created layer blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    # Forward pass
    def forward(self, x):
        """
        Docstring for forward
        
        :param self: self
        :param x: input tensor
        :return: output tensor after applying the ResNet operations
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# ResNet18 for Emotion Detection
def ResNet18_emotion(num_classes=7):
    """
    Docstring for ResNet18_emotion
    
    :param num_classes: Specifies the number of output classes for classification. Default is 7.
    :return: An instance of the ResNet model configured as ResNet18.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


# Load Trained Weights
def load_model(weights_path="trained_model/best_emotion_resnet18.pth"):
    """
    Docstring for load_model
    
    :param weights_path: Path to the trained model weights.
    :return: The ResNet18 model loaded with the specified weights.
    """
    model = ResNet18_emotion(num_classes=7)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# Predict Function
def predict_emotion(model, image_tensor, unknown_threshold=0.40):
    """
    Returns (label, confidence).
    If maximum softmax probability < unknown_threshold → returns "unknown".
    """
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)  # (1,1,48,48)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_conf = probs[0][pred_idx].item()

    # Unknown class logic
    if pred_conf < unknown_threshold:
        return "unknown", pred_conf

    return CLASSES[pred_idx], pred_conf



