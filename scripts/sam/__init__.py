import cv2
import numpy as np
import onnxruntime as ort
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

        self.processed_image = None

    @staticmethod
    def add_dummy_dim(image: np.ndarray) -> np.ndarray:
        return image[np.newaxis, :, :, :]

    def as_normalized_float_tensor(
        self, image: np.ndarray, pad_h: int, pad_w: int
    ) -> np.ndarray:
        image_f = image.copy().astype(np.float32)
        image_f -= self.mean_pix_values
        image_f /= self.std_pix_values
        image_tensor = self.add_dummy_dim(image_f).transpose((0, 3, 1, 2))
        if pad_h != 0:
            image_tensor[0, :, 0:pad_h, :] = 0  # 縦方向のパディング
            image_tensor[0, :, -pad_h:, :] = 0
        if pad_w != 0:
            image_tensor[0, :, :, 0:pad_w] = 0  # 横方向のパディング
            image_tensor[0, :, :, -pad_w:] = 0
        return image_tensor

    def resize_and_pad(
        self, target_image: np.ndarray, pad_value: int = 0
    ) -> Tuple[np.ndarray, int, int, float]:
        height, width, _ = target_image.shape
        inference_height, inference_width = self.size_for_inference

        # アスペクト比を保持しつつ目標サイズにリサイズ
        if width / height > inference_width / inference_height:
            # 横長画像の場合
            resize_rate = inference_width / width
            resized_height = int(height * resize_rate)
            resized_image = cv2.resize(target_image, (inference_width, resized_height))
            pad_top = (inference_height - resized_height) // 2
            pad_bottom = inference_height - resized_height - pad_top
            pad_left = pad_right = 0
        else:
            # 縦長画像の場合
            resize_rate = inference_height / height
            resized_width = int(width * resize_rate)
            resized_image = cv2.resize(target_image, (resized_width, inference_height))
            pad_left = (inference_width - resized_width) // 2
            pad_right = inference_width - resized_width - pad_left
            pad_top = pad_bottom = 0

        # パディングを適用
        target_image_processed = cv2.copyMakeBorder(
            resized_image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )

        return target_image_processed, pad_top, pad_left, resize_rate

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

    def generate_padded_image(self, image: np.ndarray) -> np.ndarray:
        orig_image_width, orig_image_height = image.shape[1], image.shape[0]

        # 画像のリサイズとパディング
        image_processed, pad_h, pad_w, resize_rate = self.resize_and_pad(image)

        return (
            image_processed,
            pad_h,
            pad_w,
            resize_rate,
            orig_image_height,
            orig_image_width,
        )

    def generate_image_embedding(self, image_processed, pad_h, pad_w):
        image_tensor = self.as_normalized_float_tensor(image_processed, pad_h, pad_w)
        embeddings = self.encoder.run(None, {"images": image_tensor})[0]
        return embeddings

    def mask_post_process(
        self, mask, pad_h, pad_w, orig_image_height, orig_image_width
    ):
        mask = (mask > 0).astype("uint8") * 255
        if pad_w != 0:
            mask_cropped = mask[:, pad_w:-pad_w]
        if pad_h != 0:
            mask_cropped = mask[pad_h:-pad_h]
        mask_final = cv2.resize(
            mask_cropped,
            (orig_image_width, orig_image_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return mask_final

    def decode(
        self,
        embeddings: np.ndarray,
        input_point: Tuple[int, int],
        resize_rate: float,
        longer_side_size_resized_image: int,
    ) -> np.ndarray:
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
                    [longer_side_size_resized_image, longer_side_size_resized_image],
                    dtype=np.float32,
                ),
            },
        )
        masks = outputs[0]
        return masks

    def process_image(self, image: np.ndarray, input_point: Tuple[int, int]):
        # 画像の読み込みと前処理
        (
            image_processed,
            pad_h,
            pad_w,
            resize_rate,
            orig_image_height,
            orig_image_width,
        ) = self.generate_padded_image(image)
        longer_side_size_resized_image = max(
            int(orig_image_width * resize_rate), int(orig_image_height * resize_rate)
        )

        # エンコーディング
        embeddings = self.generate_image_embedding(image_processed, pad_h, pad_w)

        # セグメンテーションの実行
        masks = self.decode(
            embeddings,
            input_point,
            resize_rate,
            longer_side_size_resized_image,
        )

        # マスクの処理
        mask = masks[0][0]
        mask_final = self.mask_post_process(
            mask, pad_h, pad_w, orig_image_height, orig_image_width
        )

        return mask_final
