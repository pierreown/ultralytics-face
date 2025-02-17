import os
from ultralytics import YOLO

params = {
    "project": "outputs/face-detect",
    "data": "datasets/wider-face.yaml",
    "epochs": 100,
    "patience": 30,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
}

pre_train = "weights/yolo11n.pt"
last_model = os.path.join(params["project"], "train", "weights", "last.pt")


def main():
    model, resume = None, None
    if not os.path.exists(last_model):
        resume = False
    else:
        resume = input("是否恢复上次训练？(y/n) [y]").strip().lower()
        resume = resume not in ["n", "no"]

    # 加载模型
    if resume:
        model = YOLO(last_model)
    else:
        model = YOLO(pre_train)

    # 训练模型
    model.train(**params, resume=resume)


if __name__ == "__main__":
    main()
