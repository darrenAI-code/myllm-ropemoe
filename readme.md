python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_moe.py --device=mps --compile=False  --batch_size=8 --block_size=128 --num_experts=2
python sample.py --out_dir=out-shakespeare-moe --device=cpu --start="ROMEO:" --num_samples=3 --max_new_tokens=200
