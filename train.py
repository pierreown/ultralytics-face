import os
from ultralytics import YOLO

params = {
    "project": "output/face-detect",
    "data": "dataset/wider-face.yaml",
    "epochs": 100,
    "imgsz": 640,
    "optimizer": "SGD",
    "workers": 12,
    "batch": 0.95,
    "device": 0,
}

pre_train = "weights/yolo11n.pt"
resume_train = os.path.join(params["project"], "train", "weights", "last.pt")


def main():
    # 加载预训练模型
    model = YOLO(resume_train)

    # 训练模型
    model.train(**params, resume=True)


if __name__ == "__main__":
    main()
