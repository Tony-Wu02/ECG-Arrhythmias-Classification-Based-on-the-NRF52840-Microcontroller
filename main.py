from concurrent.futures import ProcessPoolExecutor
import cv2
import joblib
import numpy as np
import pywt
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Initializer, LRScheduler, TensorBoard
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# Set reproducibility and deterministic mode
cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(0)

# Define the CNN module
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)     # 1 input channel, 16 output channels, 7x7 kernel
        self.conv2 = nn.Conv2d(16, 32, 3)    # 16 input, 32 output, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, 3)    # 32 input, 64 output, 3x3 kernel
        self.bn1 = nn.BatchNorm2d(16)        # Batch normalization after conv1
        self.bn2 = nn.BatchNorm2d(32)        # After conv2
        self.bn3 = nn.BatchNorm2d(64)        # After conv3
        self.pooling1 = nn.MaxPool2d(5)      # Max pooling with 5x5 window
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))  # Adaptive pooling to output 1x1
        self.fc1 = nn.Linear(68, 32)         # Fully connected: 68 in → 32 out
        self.fc2 = nn.Linear(32, 4)          # Fully connected: 32 in → 4 classes

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))      # Output shape: (16 x 94 x 94)
        x1 = self.pooling1(x1)                     # → (16 x 18 x 18)
        x1 = F.relu(self.bn2(self.conv2(x1)))      # → (32 x 16 x 16)
        x1 = self.pooling2(x1)                     # → (32 x 5 x 5)
        x1 = F.relu(self.bn3(self.conv3(x1)))      # → (64 x 3 x 3)
        x1 = self.pooling3(x1)                     # → (64 x 1 x 1)
        x1 = x1.view((-1, 64))                     # Flatten → (64,)
        x = torch.cat((x1, x2), dim=1)             # Concatenate with RR features → (68,)
        x = F.relu(self.fc1(x))                    # → (32,)
        x = self.fc2(x)                            # → (4,)
        return x


"""
Input x1 is convolved, batch normalized, and activated with a ReLU function, and the output shape is (16 x 94 x 94).
Perform max pooling, and the output shape is (16 x 18 x 18).
After the second convolution, batch normalization, and ReLU activation function, the output shape is (32 x 16 x 16).
Perform max pooling again, and the output shape is (32 x 5 x 5).
After the third convolution, batch normalization, and ReLU activation function, the output shape is (64 x 3 x 3).
Adaptive max pooling reduces the spatial dimension of each channel to (64 x 1 x 1).
Change the shape to (64,).
Concatenate x1 and x2 on dimension 1 to get a vector of shape (68,).
Pass through the first fully connected layer and apply the ReLU activation function, the output shape is (32,).
Pass through the second fully connected layer, the output shape is (4,).
Finally, the model outputs a 4-dimensional vector.
"""


def worker(data, wavelet, scales, sampling_period):
    before, after = 90, 110  # Interval around each R peak

    coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
    r_peaks, categories = data["r_peaks"], data["categories"]

    avg_rri = np.mean(np.diff(r_peaks))  # Average RR interval

    x1, x2, y, groups = [], [], [], []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue
        if categories[i] == 4:  # Skip AAMI Q class
            continue

        # Resize CWT coefficients to 100x100
        x1.append(cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
        x2.append([
            r_peaks[i] - r_peaks[i - 1] - avg_rri,                      # pre-RR
            r_peaks[i + 1] - r_peaks[i] - avg_rri,                      # post-RR
            (r_peaks[i] - r_peaks[i - 1]) / (r_peaks[i + 1] - r_peaks[i]),  # ratio-RR
            np.mean(np.diff(r_peaks[max(i - 10, 0):i + 1])) - avg_rri   # local-RR
        ])
        y.append(categories[i])
        groups.append(data["record"])

    return x1, x2, y, groups



def load_data(wavelet, scales, sampling_rate, filename="./dataset/MIT-BIH-Arrhythmia_Database/mitdb.pkl"):
    import pickle
    from sklearn.preprocessing import RobustScaler

    with open(filename, "rb") as f:
        train_data, test_data = pickle.load(f)

    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1

    # Process training set
    x1_train, x2_train, y_train, groups_train = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
            partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), train_data):
            x1_train.append(x1)
            x2_train.append(x2)
            y_train.append(y)
            groups_train.append(groups)

    x1_train = np.expand_dims(np.concatenate(x1_train), axis=1).astype(np.float32)
    x2_train = np.concatenate(x2_train).astype(np.float32)
    y_train = np.concatenate(y_train).astype(np.int64)

    # Process test set
    x1_test, x2_test, y_test, groups_test = [], [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x1, x2, y, groups in executor.map(
            partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), test_data):
            x1_test.append(x1)
            x2_test.append(x2)
            y_test.append(y)
            groups_test.append(groups)

    x1_test = np.expand_dims(np.concatenate(x1_test), axis=1).astype(np.float32)
    x2_test = np.concatenate(x2_test).astype(np.float32)
    y_test = np.concatenate(y_test).astype(np.int64)

    # Normalize RR features
    scaler = RobustScaler()
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.transform(x2_test)

    return (x1_train, x2_train, y_train, groups_train), (x1_test, x2_test, y_test, groups_test)

def main():
    sampling_rate = 360
    wavelet = "mexh"  # Available: mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101)

    (x1_train, x2_train, y_train, groups_train), (x1_test, x2_test, y_test, groups_test) = load_data(
        wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)
    print("Data loaded successfully!")

    log_dir = "./logs/{}".format(wavelet)
    shutil.rmtree(log_dir, ignore_errors=True)

    callbacks = [
        Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
        Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
        LRScheduler(policy=StepLR, step_size=5, gamma=0.1),
        EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
        TensorBoard(SummaryWriter(log_dir))
    ]

    net = NeuralNetClassifier(
        MyModule,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        max_epochs=30,
        batch_size=1024,
        train_split=predefined_split(Dataset({"x1": x1_test, "x2": x2_test}, y_test)),
        verbose=1,
        device="cpu",
        callbacks=callbacks,
        iterator_train__shuffle=True,
        optimizer__weight_decay=0,
    )

    net.fit({"x1": x1_train, "x2": x2_train}, y_train)
    y_true, y_pred = y_test, net.predict({"x1": x1_test, "x2": x2_test})

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

    net.save_params(f_params="./models/model_{}.pkl".format(wavelet))


if __name__ == "__main__":
    main()
