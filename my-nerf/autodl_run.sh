cp ~/autodl-fs/"$1".zip ~/nerf-pytorch/data/nerf_llff_data
cd ~/nerf-pytorch/data/nerf_llff_data
unzip -o "$1".zip
rm -rf "$1".zip
cd ~/nerf-pytorch/configs
cp fern.txt "$1".txt
sed -i 's/fern/'"$1"'/g' "$1".txt
cd ..
python run_nerf.py --config configs/"$1".txt