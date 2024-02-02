# =================================================================================================================
#                                           Libraries
# =================================================================================================================
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


# =================================================================================================================
#                                           Image Preparation
# =================================================================================================================
def loadImages(dataInput):
    pixelData = []
    categoryLabels = []
    imagesPathURL = []

    for folder in os.listdir(dataInput):
        folderPath = os.path.join(dataInput, folder)
        if os.path.isdir(folderPath):
            print(f"Processing -> {folderPath}")
            for image_file in os.listdir(folderPath):
                image = os.path.join(folderPath, image_file)
                convertedImage = convertTo8Gray(image)
                pixelData.append(convertedImage)
                categoryLabels.append(folder)
                imagesPathURL.append(image)

    return pixelData, categoryLabels, imagesPathURL


def LoadCustomDataSet(dataInput):
    pixelData, categoryLabels, imagesPathURL = loadImages(dataInput)
    pixelData = np.asarray(pixelData, dtype="float32") / 255.0
    categoryLabels = np.asarray(categoryLabels, dtype="int")
    imagesPathURL = np.asarray(imagesPathURL)

    xTrain, xTest, yTrain, yTest, trainPath, testPath = train_test_split(
        pixelData, categoryLabels, imagesPathURL, test_size=0.2, random_state=83
    )

    scaler = StandardScaler().fit(xTrain.reshape(-1, 64))
    xTrain = scaler.transform(xTrain.reshape(-1, 64))
    xTest = scaler.transform(xTest.reshape(-1, 64))

    return xTrain, xTest, yTrain, yTest, testPath


def convertTo8Gray(imageURL):
    image = cv2.imread(imageURL, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    return image.flatten()


# =================================================================================================================
#                                          Loading Dataset
# =================================================================================================================

xTrain, xTest, yTrain, yTest, testPath = LoadCustomDataSet("./Train")

# =================================================================================================================
#                                           Implement Your Algorithm
# =================================================================================================================

model = RandomForestClassifier(n_estimators=100, random_state=23)

model.fit(xTrain, yTrain)

predictions = model.predict(xTest)

accuracy = model.score(xTest, yTest)
print("Random Forest Model Accuracy:", accuracy * 100)

# =================================================================================================================
#                                           NEURAL NETWORK (BONUS)
# =================================================================================================================
xTrainPoint = torch.tensor(xTrain).float()
yTrainPoint = torch.tensor(yTrain).long()
xTestPoint = torch.tensor(xTest).float()
yTestPoint = torch.tensor(yTest).long()

trainDataSet = TensorDataset(xTrainPoint, yTrainPoint)
testDataSet = TensorDataset(xTestPoint, yTestPoint)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(8 * 8, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

trainLoader = DataLoader(dataset=trainDataSet, batch_size=32, shuffle=True)
testLoader = DataLoader(dataset=testDataSet, batch_size=32, shuffle=False)

for epoch in range(10):
    lossError = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lossError += loss.item()

    print(f"Epoch {epoch + 1}, Loss Value: {lossError / len(trainLoader)}")

model.eval()
predictedClasses = []
with torch.no_grad():
    correct = 0
    total = 0
    for data in testLoader:
        images, labels = data
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        predictedClasses.extend(predicted.cpu().numpy())
predicatedArray = np.array(predictedClasses)

correct = (predicatedArray == yTestPoint.numpy()).sum()
accuracy = correct / yTestPoint.size(0)
print(f"Accuracy Test Images: {accuracy * 100} %")

# =================================================================================================================
#                                           Visualization
# =================================================================================================================


testIndex = 85
image = cv2.imread(testPath[testIndex])
image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_AREA)

_lable = predictions[testIndex]
_lable_NN = predicatedArray[testIndex]

fig, ax = plt.subplots(1, 3, figsize=(30, 15))

ax[0].imshow(image, cmap=plt.cm.binary)
ax[0].set_title("Original Image")
ax[1].imshow(image, cmap=plt.cm.binary)
ax[1].set_title("RandomForestClassifier Predicted Label: {}".format(_lable))
ax[2].imshow(image, cmap=plt.cm.binary)
ax[2].set_title("Neural Network Predicted Label : {}".format(_lable_NN))
plt.show()
