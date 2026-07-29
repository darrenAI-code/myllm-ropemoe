# train a tiny MoE model on Shakespeare (uses whatever tokenizer prepare.py produced)
# run: python train.py config/train_shakespeare_moe.py
# on macbook add: --device=mps --compile=False  (or --device=cpu)
#
# The router + experts add real compute, so keep the model small when experimenting.

out_dir = 'out-shakespeare-moe'
eval_interval = 250
eval_iters = 200
log_interval = 10

# tiny dataset -> we expect to overfit fast; only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-moe'
wandb_run_name = 'mini-moe'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# small transformer backbone
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# MoE: layers 1, 3, 5 use MoE; layers 0, 2, 4 stay dense
num_experts = 4
num_experts_per_tok = 2
moe_frequency = 2         # every 2nd layer is MoE  (layer indices: 1, 3, 5)
# moe_layers = [1, 3, 5]  # equivalent explicit form; overrides moe_frequency if non-empty
use_shared_expert = True  # DeepSeek-MoE style always-active shared expert
aux_loss_coeff = 0.01     # load-balancing loss weight

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# device = 'mps'
# compile = False
