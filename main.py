import cv2
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, List
from copy import deepcopy


class SegmentationAnythingModule:
    def __init__(self, enc_model_path: str, sam_model_path: str):
        self.enc_model_path = enc_model_path
        self.sam_model_path = sam_model_path

        self.encoder = ort.InferenceSession(
            self.enc_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            self.sam_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.mean_pix_values = np.array([123.675, 116.28, 103.53])
        self.std_pix_values = np.array([[58.395, 57.12, 57.375]])
        self.size_for_inference: Tuple[int, int] = (1024, 1024)

    @staticmethod
    def add_dummy_dim(image: np.ndarray) -> np.ndarray:
        return image[np.newaxis, :, :, :]

    def as_normalized_float_tensor(self, image: np.ndarray, pad_h: int) -> np.ndarray:
        image_f = image.copy().astype(np.float32)
        image_f -= self.mean_pix_values
        image_f /= self.std_pix_values
        image_tensor = self.add_dummy_dim(image_f).transpose((0, 3, 1, 2))
        image_tensor[0, :, 0:pad_h, :] = 0
        image_tensor[0, :, -pad_h:, :] = 0
        return image_tensor

    def resize_and_pad(
        self, target_image: np.ndarray, pad_value: int = 0
    ) -> Tuple[np.ndarray, int, float]:
        _, image_width_raw, _ = target_image.shape
        resize_rate = float(self.size_for_inference[0]) / image_width_raw
        target_image_resized = cv2.resize(
            target_image, None, fx=resize_rate, fy=resize_rate
        )
        image_height_resized, _, _ = target_image_resized.shape

        pad_h = (self.size_for_inference[1] - image_height_resized) // 2
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
        target_image_processed = cv2.resize(
            target_image_processed, self.size_for_inference
        )
        return target_image_processed, pad_h, resize_rate

    def generate_coords_and_labels_from_point(
        self, input_point_xy: List[int], resize_rate_from_orig: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        input_point = np.array([input_point_xy])
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

    def process_image(self, image: np.ndarray, input_point: Tuple[int, int]):
        # 画像の読み込みと前処理
        orig_image_width, orig_image_height = image.shape[1], image.shape[0]

        # 画像のリサイズとパディング
        image_processed, pad_h, resize_rate = self.resize_and_pad(image)
        resized_image_width = int(orig_image_width * resize_rate)

        # エンコーディング
        image_tensor = self.as_normalized_float_tensor(image_processed, pad_h)
        embeddings = self.encoder.run(None, {"images": image_tensor})[0]

        onnx_coord, onnx_label = self.generate_coords_and_labels_from_point(
            list(input_point), resize_rate
        )

        # セグメンテーションの実行
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        outputs = self.decoder.run(
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
        mask_final = cv2.resize(
            mask_cropped,
            (orig_image_width, orig_image_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask_final


def main(image_path, enc_model_path, sam_model_path, input_point):
    # 画像の読み込みと前処理
    image = cv2.imread(image_path)

    sam_module = SegmentationAnythingModule(enc_model_path, sam_model_path)
    mask = sam_module.process_image(image, input_point)

    # オーバーレイの作成
    mask_rgb = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 1
    where_nonzero = np.where(mask > 0)
    mask_rgb[where_nonzero[0], where_nonzero[1], :] = np.array([0, 255, 0])

    result = cv2.addWeighted(image, 0.6, mask_rgb, 0.4, 2.2)

    # 結果の表示
    cv2.imshow("Segmentation Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
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
        default="960,810",
        help="Input point for segmentation in format x,y (e.g., 321,230)",
    )

    # 引数を解析
    args = parser.parse_args()

    image_path = str(Path(args.image_path))
    enc_model_path = str(Path(args.enc_model_path))
    sam_model_path = str(Path(args.sam_model_path))
    input_points = tuple(map(int, args.input_point.split(",")))

    main(image_path, enc_model_path, sam_model_path, input_points)
