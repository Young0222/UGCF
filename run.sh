
# ml-1m
# first stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="mf" --decay=1e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="ml-1m" --topks="[20]" --recdim=32 --path="ckpt-ml1m-mf" --epochs=200 --alpha=1e-2
# second stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="ours" --decay=5e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="ml-1m" --topks="[20]" --recdim=64 --path="ckpt-ml1m-lgn" --epochs=1000 --alpha=1e-2

# amazon-book
# first stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="mf" --decay=1e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="amazon-book" --topks="[20]" --recdim=64 --path="ckpt-amazon-mf" --epochs=200 --alpha=1e-2
# second stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="ours" --decay=1e-4 --lr=0.005 --layer=3 --seed=2023 --dataset="amazon-book" --topks="[20]" --recdim=64 --path="ckpt-amazon-lgn" --epochs=200 --alpha=1e-2

# yelp
# first stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="mf" --decay=1e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="yelp" --topks="[20]" --recdim=32 --path="ckpt-yelp-mf" --alpha=1e-2
# second stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="ours" --decay=1e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="yelp" --topks="[20]" --recdim=64 --path="ckpt-yelp-lgn" --epochs=200 --alpha=1e-2

# synthetic
# first stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="mf" --decay=1e-4 --lr=0.01 --layer=3 --seed=2023 --dataset="syn" --topks="[20]" --recdim=32 --path="ckpt-syn-mf" --epochs=200 --testbatch=64 --alpha=1e-2
# second stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="ours" --decay=1e-4 --lr=0.01 --layer=3 --seed=2023 --dataset="syn" --topks="[20]" --recdim=64 --path="ckpt-syn-lgn" --epochs=200 --testbatch=64 --alpha=1e-2