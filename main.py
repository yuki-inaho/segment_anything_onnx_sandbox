import cv2
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple
from copy import deepcopy


MEAN_PIX_VALUES = np.array([123.675, 116.28, 103.53])
STD_PIX_VALUES = np.array([[58.395, 57.12, 57.375]])
SIZE_FOR_INFERENCE = (1024, 1024)

# 引数からパラメータを設定
INPUT_POINT = (960, 810)


def add_dummy_dim(image: np.ndarray) -> np.ndarray:
    return image[np.newaxis, :, :, :]


def as_normalized_float_tensor(image, pad_h, mean_pix_norm, std_pix_norm):
    image_f = image.copy().astype(np.float32)
    image_f -= np.array(mean_pix_norm)
    image_f /= np.array(std_pix_norm)
    image_tensor = add_dummy_dim(image_f).transpose((0, 3, 1, 2))
    image_tensor[0, :, 0:pad_h, :] = 0
    image_tensor[0, :, -pad_h:, :] = 0
    return image_tensor


def resize_and_pad(
    target_image: np.ndarray, image_size_inference: Tuple[int, int], pad_value: int = 0
):
    _, image_width_raw, _ = target_image.shape
    resize_rate = float(image_size_inference[0]) / image_width_raw
    target_image_resized = cv2.resize(
        target_image, None, fx=resize_rate, fy=resize_rate
    )
    image_height_resized, _, _ = target_image_resized.shape

    pad_h = (image_size_inference[1] - image_height_resized) // 2
    if pad_h > 0:
        target_image_processed = cv2.copyMakeBorder(
            target_image_resized,
            pad_h,
            pad_h,
            0,
            0,
            cv2.BORDER_CONSTANT,
            (pad_value, pad_value, pad_value),
        )
    else:
        target_image_processed = target_image_resized
    target_image_processed = cv2.resize(target_image_processed, image_size_inference)
    return target_image_processed, pad_h, resize_rate


def generate_coords_and_labels_from_point(
    input_point_x, input_point_y, resize_rate_from_orig
):
    input_point = np.array([[input_point_x, input_point_y]])
    input_label = np.array([1])

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[
        None, :, :
    ]
    onnx_label = np.concatenate([input_label, np.array([-1])])[None, :].astype(
        np.float32
    )

    coords = deepcopy(onnx_coord).astype(float)
    coords[..., 0] = coords[..., 0] * resize_rate_from_orig
    coords[..., 1] = coords[..., 1] * resize_rate_from_orig

    onnx_coord = coords.astype("float32")
    return onnx_coord, onnx_label


# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="Image Segmentation Script")
parser.add_argument(
    "-i",
    "--image_path",
    type=str,
    default="data/apple_5x4.jpg",
    help="Path to the image",
)
parser.add_argument(
    "--enc_model_path",
    type=str,
    default="models/image_encoder_vit_b.onnx",
    help="Path to the encoder model",
)
parser.add_argument(
    "--sam_model_path",
    type=str,
    default="models/sam_vit_b_01ec64.onnx",
    help="Path to the SAM model",
)
parser.add_argument(
    "--input_point",
    type=str,
    default="321,230",
    help="Input point for segmentation in format x,y (e.g., 321,230)",
)

# 引数を解析
args = parser.parse_args()

input_points = tuple(map(int, args.input_point.split(",")))

# 画像の読み込みと前処理
image = cv2.imread(args.image_path)
orig_image_width, orig_image_height = image.shape[1], image.shape[0]


# 画像のリサイズとパディング
image_processed, pad_h, resize_rate = resize_and_pad(image, SIZE_FOR_INFERENCE)
resized_image_width, resized_image_height = int(orig_image_width * resize_rate), int(
    orig_image_height * resize_rate
)

# エンコーダモデルのロード
encoder = ort.InferenceSession(
    args.enc_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
image_tensor = as_normalized_float_tensor(
    image_processed, pad_h, MEAN_PIX_VALUES, STD_PIX_VALUES
)

# エンコーディング
embeddings = encoder.run(None, {"images": image_tensor})[0]

onnx_coord, onnx_label = generate_coords_and_labels_from_point(
    INPUT_POINT[0], INPUT_POINT[1], resize_rate
)

# デコーダモデルのロード
decoder = ort.InferenceSession(
    args.sam_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# セグメンテーションの実行
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)
outputs = decoder.run(
    None,
    {
        "image_embeddings": embeddings,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(
            [resized_image_width, resized_image_width], dtype=np.float32
        ),
    },
)
masks = outputs[0]

# マスクの処理
mask = masks[0][0]
mask = (mask > 0).astype("uint8") * 255
mask_cropped = mask[pad_h:-pad_h]
mask_rawsize = cv2.resize(
    mask_cropped, (orig_image_width, orig_image_height), interpolation=cv2.INTER_NEAREST
)

# オーバーレイの作成
mask_rgb = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 1
where_nonzero = np.where(mask_rawsize > 0)
mask_rgb[where_nonzero[0], where_nonzero[1], :] = np.array([0, 255, 0])

result = cv2.addWeighted(image, 0.6, mask_rgb, 0.4, 2.2)

# 結果の表示
cv2.imshow("Segmentation Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
