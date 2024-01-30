import cv2
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, List
from copy import deepcopy
from scripts.sam import SegmentationAnythingModule


def main(
    image_path, enc_model_path, sam_model_path, input_point, resize_rate_to_visualize
):
    # 画像の読み込みと前処理
    image = cv2.imread(image_path)

    sam_module = SegmentationAnythingModule(enc_model_path, sam_model_path)
    mask = sam_module.process_image(image, input_point)

    # オーバーレイの作成
    mask_rgb = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 1
    where_nonzero = np.where(mask > 0)
    mask_rgb[where_nonzero[0], where_nonzero[1], :] = np.array([0, 255, 0])

    result = cv2.addWeighted(image, 0.6, mask_rgb, 0.4, 2.2)
    result_to_visualize = cv2.resize(
        result,
        (
            int(result.shape[1] * resize_rate_to_visualize),
            int(result.shape[0] * resize_rate_to_visualize),
        ),
    )

    # 結果の表示
    cv2.imshow("Segmentation Result", result_to_visualize)
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
        "-em",
        "--enc_model_path",
        type=str,
        default="models/image_encoder_vit_b.onnx",
        help="Path to the encoder model",
    )
    parser.add_argument(
        "-sm",
        "--sam_model_path",
        type=str,
        default="models/sam_vit_b_01ec64.onnx",
        help="Path to the SAM model",
    )
    parser.add_argument(
        "-ip",
        "--input_point",
        type=str,
        default="960,810",
        help="Input point for segmentation in format x,y (e.g., 321,230)",
    )
    parser.add_argument(
        "--resize-rate-to-visualize",
        type=float,
        default=0.5,
        help="Resize rate to visualize the result",
    )

    # 引数を解析
    args = parser.parse_args()

    image_path = str(Path(args.image_path))
    enc_model_path = str(Path(args.enc_model_path))
    sam_model_path = str(Path(args.sam_model_path))
    input_points = tuple(map(int, args.input_point.split(",")))
    resize_rate_to_visualize = args.resize_rate_to_visualize

    main(
        image_path,
        enc_model_path,
        sam_model_path,
        input_points,
        resize_rate_to_visualize,
    )
