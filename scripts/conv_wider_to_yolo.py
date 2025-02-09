import os
from PIL import Image
from tqdm import tqdm


def get_image_size(image_path: str):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None


# 解析 WIDER 数据集中的图片与标签
def parse_wider_split(split_path: str):
    line_type, img_path, label_count, label_index = 0, None, 0, 0
    labels = []  # 存储每张图片的标注信息
    with open(split_path, "r", encoding="utf-8", buffering=4096) as split_file:
        for line in split_file:
            line = line.strip()
            if line_type == 0:
                # 读取图片路径
                img_path = line
                line_type = 1
            elif line_type == 1:
                # 读取标签数量
                label_count = int(line)
                line_type = 2
            elif line_type == 2:
                label_index += 1
                if label_index <= label_count:
                    # 将每个标签的各个属性拆分出来并存入字典
                    tags = line.split(" ")
                    labels.append(
                        {
                            "x": int(tags[0]),
                            "y": int(tags[1]),
                            "w": int(tags[2]),
                            "h": int(tags[3]),
                            "blur": int(tags[4]),
                            "expression": int(tags[5]),
                            "illumination": int(tags[6]),
                            "invalid": int(tags[7]),
                            "occlusion": int(tags[8]),
                            "pose": int(tags[9]),
                        }
                    )

                # 标签读取完毕，准备返回该图片及其标签
                if label_index < label_count:
                    continue

                yield img_path, labels

                # 清空当前图片的标签并准备读取下一张
                labels, line_type, label_index = [], 0, 0


# 将 WIDER 数据集转换为 YOLO 格式
def conv_wider_to_yolo(splits: dict, images: dict, data_root: str):
    for split_name, split_path in splits.items():
        # 设置目标图像存储路径
        image_root = os.path.join(data_root, "images", split_name)
        if not os.path.exists(image_root):
            os.makedirs(image_root)

        # 设置目标标签存储路径
        label_root = os.path.join(data_root, "labels", split_name)
        if not os.path.exists(label_root):
            os.makedirs(label_root)

        # 遍历每张图片的路径及其标签，进行数据转换
        for image_path, labels in parse_wider_split(split_path):
            # 获取图像的绝对路径
            image_abs_path = os.path.join(images[split_name], image_path)
            # 获取图像文件名
            image_name = os.path.basename(image_path)
            # 标签文件路径
            label_path = os.path.splitext(image_name)[0]
            label_path = os.path.join(label_root, label_path + ".txt")
            # 图像目标路径
            image_tgt_path = os.path.join(image_root, image_name)

            # 获取图像的宽和高
            img_size = get_image_size(image_abs_path)
            if img_size is None:
                continue
            img_w, img_h = img_size

            has_label = False  # 标志，表示该图片是否有标签
            with open(label_path, "w", encoding="utf-8") as f:
                for label in labels:
                    # 归一化边界框坐标
                    sx, sy, box_w, box_h = (
                        float(label["x"]),
                        float(label["y"]),
                        float(label["w"]),
                        float(label["h"]),
                    )
                    sx = min(max(sx, 0), img_w)
                    sy = min(max(sy, 0), img_h)
                    ex = min(max(sx + box_w, 0), img_w)
                    ey = min(max(sy + box_h, 0), img_h)
                    box_w, box_h = ex - sx, ey - sy

                    w, h = box_w / img_w, box_h / img_h

                    # 如果目标过小，则跳过该目标
                    if w < 0.05 or h < 0.05:
                        continue

                    # 将边界框转换为YOLO格式，并写入标签文件
                    x, y = (sx + box_w / 2) / img_w, (sy + box_h / 2) / img_h
                    f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(0, x, y, w, h))
                    has_label = True

            # 如果没有标签，则删除标签文件
            if not has_label:
                os.remove(label_path)
                continue

            # 创建图片符号链接
            if not os.path.exists(image_tgt_path):
                os.symlink(image_abs_path, image_tgt_path)


if __name__ == "__main__":
    src_root = "D:/AI/Datasets/WIDER_FACE/ORIG"
    dst_root = "D:/AI/Datasets/WIDER_FACE/YOLO"

    wider_splits = {
        "train": os.path.join(src_root, "split", "wider_face_train_bbx_gt.txt"),
        "val": os.path.join(src_root, "split", "wider_face_val_bbx_gt.txt"),
    }
    wider_images = {
        "train": os.path.join(src_root, "train"),
        "val": os.path.join(src_root, "val"),
    }

    conv_wider_to_yolo(wider_splits, wider_images, dst_root)
