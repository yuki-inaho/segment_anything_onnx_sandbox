# Related documents

- https://dev.to/andreygermanov/export-segment-anything-neural-network-to-onnx-the-missing-parts-43c8
- https://github.com/facebookresearch/segment-anything/issues/16

## Download a SAM model

```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b_01ec64.pth
```

## Install dependencies

```
pip install -r requirements.txt
cd segment-anything; pip install -e .
```

## Convert the pytorch model to onnx model (image encoder)

```
python export_image_encoder.py \
-c models/sam_vit_b_01ec64.pth \
-m vit_b \
-o models/image_encoder_vit_b.onnx
```

## Convert the pytorch model to onnx model ()

```
python segment-anything/scripts/export_onnx_model.py \
--checkpoint models/sam_vit_b_01ec64.pth \
--model-type vit_b \
--output models/sam_vit_b_01ec64.onnx \
--return-single-mask
```
