import torch

default_model_path = "outputs/face-detect/train/weights/last.pt"

model_path = input("请输入模型文件路径: [" + default_model_path + "]").strip()

if not model_path:
    model_path = default_model_path

model = torch.load(model_path, weights_only=False)

model["train_args"].update(
    {
        "model": "outputs/face-detect/train/weights/last.pt",
        "data": "datasets/wider-face.yaml",
        "project": "outputs/face-detect",
        "resume": "outputs/face-detect/train/weights/last.pt",
        "save_dir": "outputs/face-detect/train",
    }
)

torch.save(model, model_path + ".fix")
