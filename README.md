<img width="720" alt="sam result" src="https://github.com/yuki-inaho/segment_anything_onnx_sandbox/blob/main/doc/result.jpg">

# Related documents

- https://dev.to/andreygermanov/export-segment-anything-neural-network-to-onnx-the-missing-parts-43c8
- https://github.com/facebookresearch/segment-anything/issues/16

# Workflow

## Download a SAM model

- https://github.com/facebookresearch/segment-anything

```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b_01ec64.pth
```

## Install dependencies (under the CUDA v11.8 environment)

```
git submodule --init --recursive \
cd segment-anything \
pip install -r requirements.txt \
pip install -e .
```

## Convert the pytorch model to onnx model (image encoder)

```
python export_image_encoder.py \
-c models/sam_vit_b_01ec64.pth \
-m vit_b \
-o models/image_encoder_vit_b.onnx
```

## Convert the pytorch model to onnx model (mask decoder)

```
python segment-anything/scripts/export_onnx_model.py \
--checkpoint models/sam_vit_b_01ec64.pth \
--model-type vit_b \
--output models/sam_vit_b_01ec64.onnx \
--return-single-mask
```

## Run some processes

```
python main.py \
-i data/apple_5x4.jpg \
-ip 960,810
```

```
python main.py \
-i data/crab.png \
-ip 384,512
```
