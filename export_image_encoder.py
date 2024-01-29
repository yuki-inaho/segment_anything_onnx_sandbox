import argparse
import torch
from segment_anything import sam_model_registry


def export_to_onnx(model_variant, checkpoint_path, output_path):
    sam = sam_model_registry[model_variant](checkpoint=checkpoint_path)

    # Export images encoder from SAM model to ONNX
    torch.onnx.export(
        model=sam.image_encoder,
        args=torch.randn(1, 3, 1024, 1024),
        f=output_path,
        input_names=["images"],
        output_names=["embeddings"],
        export_params=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM model's image encoder to ONNX format."
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "-m",
        "--model_variant",
        type=str,
        choices=["vit_h", "vit_l", "vit_b"],
        required=True,
        help="Model variant to use (vit_h, vit_l, vit_b).",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Output path for the ONNX model.",
    )

    args = parser.parse_args()
    export_to_onnx(
        model_variant=args.model_variant,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
