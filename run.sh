
# ml-1m
# first stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="mf" --decay=1e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="ml-1m" --topks="[20]" --recdim=32 --path="ckpt-ml1m-ours-mf" --epochs=100
# second stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="ours" --decay=5e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="ml-1m" --topks="[20]" --recdim=64 --path="ckpt-ml1m-ours-lgn" --epochs=100

# amazon-book
# first stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="mf" --decay=1e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="amazon-book" --topks="[20]" --recdim=64 --path="ckpt-amazon-ours-mf" --epochs=100
# second stage
CUDA_VISIBLE_DEVICES=1 python main.py --model="ours" --decay=1e-4 --lr=0.005 --layer=3 --seed=2023 --dataset="amazon-book" --topks="[20]" --recdim=64 --path="ckpt-amazon-ours-lgn" --epochs=100

# synthetic dataset
# first stage
CUDA_VISIBLE_DEVICES=5 python main.py --model="mf" --decay=1e-4 --lr=0.001 --layer=3 --seed=2023 --dataset="syn" --topks="[20]" --recdim=32 --path="ckpt-syn-ours-mf" --epochs=100 --testbatch 128
# second stage
CUDA_VISIBLE_DEVICES=5 python main.py --model="ours" --decay=1e-4 --lr=0.01 --layer=3 --seed=2023 --dataset="syn" --topks="[20]" --recdim=64 --path="ckpt-syn-ours-lgn" --epochs=100 --testbatch 128