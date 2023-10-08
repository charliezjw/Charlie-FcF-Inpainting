python train.py \
    --outdir=training-runs-inp \
    --img_data=/data/charlie/places2_dataset/train \
    --gpus 8 \
    --kimg 25000 \
    --gamma 10 \
    --aug 'noaug' \
    --metrics True \
    --eval_img_data /data/charlie/places2_dataset/evaluation/random_segm_256
    --batch 128

python train.py \
  --outdir=training-runs-inp \
  --img_data=/data/charlie/places2_dataset/train \
  --gpus 8 \
  --kimg 25000 \
  --gamma 10 \
  --aug 'noaug' \
  --metrics True \
  --eval_img_data /data/charlie/places2_dataset/evaluation/random_segm_256 \
  --batch 128 \
  --resolution 512