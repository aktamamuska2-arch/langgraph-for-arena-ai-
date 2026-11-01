"""Konfigurasi P1-sandbox (PREREGISTRATION v1.3).

Semua angka di sini berasal dari dokumen terkunci, bukan selera saat menulis kode:
  - profil model & anggaran token: PREREG §2, §15
  - seed baseline & seed rantai: PREREG §2 (101, 202, 303, 404, 505)
  - N_syn = 0,30 |T0|: PREREG §2.2
  - 6 generasi, 3 arm: PREREG §2
"""

PROFILE = "P1S"

SEED_GLOBAL = 20261101
BASELINE_SEEDS = [101, 202, 303, 404, 505]
CHAIN_SEEDS = [101, 202, 303, 404, 505]
ARMS = ["A", "B1", "C"]
GENERATIONS = 6
N_SYN_RATIO = 0.30

# model (PREREG v1.3: 969.472 parameter, 6 lapis, konteks 256)
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 4
D_MODEL = 112

# pelatihan
BATCH_SIZE = 32
EPOCHS = 1.0          # anggaran token = 1 x |T0-latih|, sama untuk semua model
LR = 1e-3
MIN_LR = 1e-4
WARMUP = 20
GRAD_CLIP = 1.0

# generasi
TEMPERATURE = 0.9
TOP_P = 0.95
GEN_MAX_NEW = 240
GEN_BATCH = 64
GEN_MIN_DOC = 40      # dokumen lebih pendek dibuang sebelum dipakai melatih

# evaluasi (PREREG §6.2: >= 100 dokumen, >= 40 item aritmetika)
EVAL_DOCS = 108
ARITH_N = 60
ENTROPY_PROMPTS = 20
ENTROPY_SAMPLES = 5
ENTROPY_TOKENS = 12
NEAR_DUP_N = 5
NEAR_DUP_JACCARD = 0.8
ANCHOR_BYTES = 160_000


def seed_for(seed: int, gen: int) -> int:
    """Seed pelatihan generasi g. Deterministik, sama untuk ketiga arm (berpasangan)."""
    return seed + 7919 * gen
