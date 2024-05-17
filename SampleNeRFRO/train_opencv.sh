SCENE=glass
EXPERIMENT=refractive-nerf-jax
TRAIN_DIR=~/autodl-tmp/SampleNeRFRO/logs/$EXPERIMENT/${SCENE}
DATA_DIR=~/autodl-tmp/SampleNeRFRO/data/$SCENE

python -m train \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --config=configs/"${SCENE}" \
  --gin_file=configs/${SCENE}.gin \
  --stage="radiance_pe-bkgd_bg-smooth-l2-1.0-ps-128_bbox-1.0_wo-U"
