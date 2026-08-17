python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_moe.py --device=mps --compile=False  --batch_size=8 --block_size=128 --num_experts=4
python sample.py --out_dir=out-shakespeare-moe --device=cpu --start="ROMEO:" --num_samples=3 --max_new_tokens=128
python train_muon.py config/train_openwebtext_moe.py --device=mps --compile=True  --batch_size=16 --block_size=256 --num_experts=4
python sample.py --out_dir=out-openwebtext-moe  --device=cpu --start="What we do you say is that everyone" --num_samples=3 --max_new_tokens=256

