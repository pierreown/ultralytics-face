import os
import torch

model_path = input("请输入模型文件路径: ").strip()

if not model_path:
    print("模型文件路径不能为空")
    exit(1)

if not os.path.exists(model_path):
    print("模型文件不存在")
    exit(1)

model = torch.load(model_path, weights_only=False)

model["train_args"].update(
    {
        # "model": "outputs/face-detect/train/weights/last.pt",
        # "data": "datasets/wider-face.yaml",
        # "project": "outputs/face-detect",
        # "resume": "outputs/face-detect/train/weights/last.pt",
        # "save_dir": "outputs/face-detect/train",
    }
)

torch.save(model, model_path + ".fix")
