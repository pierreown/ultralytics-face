import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


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


# def stat_wider_face(split_path: str, image_root: str):
#     """统计 WIDER FACE 数据集目标框的尺寸分布，并可视化"""
#     metas = []
#     ignored_images = []

#     aspect_ratios = []  # 存储宽高比 w/h
#     areas = []  # 存储目标框面积 (w * h)
#     blur_counts = collections.Counter()  # 统计模糊程度
#     illum_counts = collections.Counter()  # 统计光照情况
#     occlusion_counts = collections.Counter()  # 统计遮挡程度

#     # 统计目标框尺寸的分布
#     width_hist = collections.Counter()
#     height_hist = collections.Counter()

#     # 遍历所有图片
#     for image_path, labels in tqdm(
#         parse_wider_split(split_path), desc="Processing images"
#     ):
#         image_abs_path = os.path.join(image_root, image_path)

#         # 获取图像尺寸
#         img_size = get_image_size(image_abs_path)
#         if img_size is None:
#             ignored_images.append(image_abs_path)
#             continue

#         img_w, img_h = img_size

#         for label in labels:
#             # 归一化目标框坐标，确保边界不超出图像范围
#             sx, sy, box_w, box_h = (
#                 max(0, min(float(label["x"]), img_w)),
#                 max(0, min(float(label["y"]), img_h)),
#                 max(0, min(float(label["w"]), img_w)),
#                 max(0, min(float(label["h"]), img_h)),
#             )

#             ex = min(sx + box_w, img_w)
#             ey = min(sy + box_h, img_h)
#             box_w, box_h = ex - sx, ey - sy

#             # 计算归一化尺寸
#             norm_w, norm_h = box_w / img_w, box_h / img_h
#             metas.append((norm_w, norm_h))

#             # 统计宽高比和面积
#             if box_h > 0:  # 避免除零错误
#                 aspect_ratios.append(box_w / box_h)
#             areas.append(norm_w * norm_h)

#             # 统计尺寸分布（四舍五入到小数点后三位）
#             width_hist[round(norm_w, 3)] += 1
#             height_hist[round(norm_h, 3)] += 1

#             # 统计标签属性
#             blur_counts[label["blur"]] += 1
#             illum_counts[label["illumination"]] += 1
#             occlusion_counts[label["occlusion"]] += 1

#     print(f"\n[INFO] 统计完成，共解析 {len(metas)} 个目标框")
#     if ignored_images:
#         print(
#             f"[WARNING] {len(ignored_images)} 张图片无法读取，建议检查: {ignored_images[:5]}"
#         )

#     # 计算统计数据
#     if metas:
#         mean_w, mean_h = np.mean(metas, axis=0)
#         min_w, min_h = np.min(metas, axis=0)
#         max_w, max_h = np.max(metas, axis=0)

#         print(f"[INFO] 平均目标框宽度: {mean_w:.4f}, 高度: {mean_h:.4f}")
#         print(f"[INFO] 最小目标框: ({min_w:.4f}, {min_h:.4f})")
#         print(f"[INFO] 最大目标框: ({max_w:.4f}, {max_h:.4f})")
#         print(f"[INFO] 平均宽高比 (w/h): {np.mean(aspect_ratios):.4f}")
#         print(f"[INFO] 目标框面积均值: {np.mean(areas):.4f}")

#     # 绘制目标框的宽度和高度分布
#     plot_distribution(width_hist, height_hist)

#     # 绘制属性统计
#     plot_attributes(blur_counts, illum_counts, occlusion_counts)


# def plot_distribution(width_hist, height_hist):
#     """绘制目标框的宽度和高度分布直方图"""
#     fig, axs = plt.subplots(1, 2, figsize=(12, 5))

#     axs[0].bar(
#         width_hist.keys(), width_hist.values(), width=0.005, color="b", alpha=0.7
#     )
#     axs[0].set_title("目标框宽度分布")
#     axs[0].set_xlabel("归一化宽度 (w)")
#     axs[0].set_ylabel("频数")

#     axs[1].bar(
#         height_hist.keys(), height_hist.values(), width=0.005, color="g", alpha=0.7
#     )
#     axs[1].set_title("目标框高度分布")
#     axs[1].set_xlabel("归一化高度 (h)")
#     axs[1].set_ylabel("频数")

#     plt.tight_layout()
#     plt.show()


def stat_wider_face(split_path: str, image_root: str):
    """Count blur, illumination, invalid, occlusion attributes."""
    blur_counts = defaultdict(int)
    illum_counts = defaultdict(int)
    invalid_counts = defaultdict(int)
    occlusion_counts = defaultdict(int)

    for image_path, labels in tqdm(parse_wider_split(split_path)):
        image_abs_path = os.path.join(image_root, image_path)

        # Check if the image exists
        if get_image_size(image_abs_path) is None:
            continue

        for label in labels:
            blur_counts[label["blur"]] += 1
            illum_counts[label["illumination"]] += 1
            invalid_counts[label["invalid"]] += 1
            occlusion_counts[label["occlusion"]] += 1

    return blur_counts, illum_counts, invalid_counts, occlusion_counts


def plot_attributes(blur_counts, illum_counts, invalid_counts, occlusion_counts):
    """Plot bar charts for blur, illumination, invalid, and occlusion statistics."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    attributes = [
        ("Blur Level", blur_counts, ["Clear", "Slightly", "Severely"]),
        ("Illumination", illum_counts, ["Normal", "Extreme"]),
        ("Invalid Faces", invalid_counts, ["Valid", "Invalid"]),
        ("Occlusion", occlusion_counts, ["None", "Partial", "Severe"]),
    ]

    for i, (title, counts, labels) in enumerate(attributes):
        keys = sorted(counts.keys())
        values = [counts.get(k, 0) for k in keys]
        total_count, max_value = sum(values), max(values)
        percentages = [(v / total_count) * 100 for v in values]  # 转换为百分比

        ax = axs[i // 2, i % 2]

        ax.bar(keys, percentages, color="c", alpha=0.7, width=0.6)
        ax.set_title(title)

        ax.set_xticks(keys)
        ax.set_xticklabels(labels)

        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)  # 0% - 100% 范围
        ax.set_yticks(range(0, 101, 10))  # 0%、20%、40%...

        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # 添加数值标签
        for k, v, p in zip(keys, values, percentages):
            if p < 15:
                ax.text(k, p + 2, f"{v}\n({p:.2f}%)", ha="center", va="bottom")
            else:
                ax.text(k, p - 2, f"{v}\n({p:.2f}%)", ha="center", va="top")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_root = "D:/AI/Datasets/WIDER_FACE/ORIG/train"
    split_file = "D:/AI/Datasets/WIDER_FACE/ORIG/split/wider_face_train_bbx_gt.txt"

    blur_counts, illum_counts, invalid_counts, occlusion_counts = stat_wider_face(
        split_file, image_root
    )
    plot_attributes(blur_counts, illum_counts, invalid_counts, occlusion_counts)
