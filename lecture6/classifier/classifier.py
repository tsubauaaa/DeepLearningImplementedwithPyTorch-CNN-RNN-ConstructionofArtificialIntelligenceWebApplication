import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, Markup, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = [
    "australia-brick",
    "basic-brick",
    "hanmasu-red-brick",
    "hanpen-red-brick",
    "normal-red-brick",
    "saboten-brick",
]
n_class = len(labels)
img_size = 224
n_result = 6  # 上位6つの結果を表示

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=28 * 28 * 128, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        # ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index"))

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像の読み込み
        image = Image.open(filepath)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))

        normalize = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )  # 平均値を0、標準偏差を1に
        to_tensor = transforms.ToTensor()
        transform = transforms.Compose([to_tensor, normalize])

        x = transform(image)
        x = x.reshape(1, 3, img_size, img_size)

        # 予測
        net = Net(n_class)
        net.load_state_dict(
            torch.load("model_cnn.pth", map_location=torch.device("cpu"))
        )
        net.eval()  # 評価モード

        y = net(x)
        y_pred = torch.argmax(y, dim=1)
        result = "<p>" + "このレンガは" + str(labels[y_pred]) + "です" + "</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath)
        # result = ""
        # for i in range(n_result):
        #     idx = sorted_idx[i].item()
        #     ratio = y[idx].item()
        #     label = labels[idx]
        #     result += "<p>" + str(round(ratio * 100, 1)) + "%の確率で" + label + "です。</p>"
        # return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
