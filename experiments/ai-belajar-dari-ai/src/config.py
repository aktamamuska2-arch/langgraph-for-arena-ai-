"""Konfigurasi profil P0 (sandbox: 2 vCPU, 3 GB, tanpa GPU)."""

PROFILE = "P0"
SEED_GLOBAL = 20261101
BASELINE_SEEDS = [13, 1337, 2024, 7]
CHAIN_SEEDS = [13, 1337]
ARMS = ["A", "B1"]
GENERATIONS = 3

T0_BYTES = 200_000
GEN_BYTES = 60_000          # anggaran korpus sintetis per generasi (dibatasi wall-clock)
TRAIN_STEPS = 600
BATCH_SIZE = 16
LR = 1e-3
MIN_LR = 1e-4
WARMUP = 20
GRAD_CLIP = 1.0

# model (harus sama dengan src/model.py XSConfig)
BLOCK_SIZE = 96
N_LAYER = 4
N_HEAD = 4
D_MODEL = 96

# generasi
GEN_BATCH = 32
GEN_MAX_NEW = 80
TEMPERATURE = 0.9
TOP_P = 0.95
GEN_CATALOG_LIMIT = 2000

# filter
MIN_DOC_BYTES = 40
MAX_DOC_BYTES = 600

# evaluasi
EVAL_SAMPLES = 24
EVAL_ARITH_N = 12

# gate (PLAN v0.3 §10.5, aturan R7)
N_LOOKS = 180
FWER = 0.05
FLOOR_BPB = 0.03
