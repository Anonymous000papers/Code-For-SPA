# SPA: Structure-Preserving Approximation of Gated Activations under FHE

Code and experiment harnesses for the paper *SPA: Structure-Preserving
Approximation of Gated Activations under FHE*.

SPA approximates a complementary-gated activation `A(x)` (GELU, SiLU) by
exploiting the identity `A(x) = x/2 + H(x²)` with `H` even: a Chebyshev
minimax series of degree `k` on the square domain `t = 2x²/R² − 1` gives a
degree-`2k` approximation in `x` at the cost of a degree-`k` evaluation.
The repository contains three components:

| directory | what it does | model / library |
|---|---|---|
| `SPA_GELU/` | plaintext evaluation: replaces the GELU activation of a decoder-only LLM by SPA and by comparison methods (MOAI, NEXUS, Euston) and measures downstream accuracy and perplexity on ten benchmarks | Gemma 4 E2B (`gelu_pytorch_tanh`, or the exact erf-GELU substituted as reference); PyTorch, transformers |
| `SPA_SiLU/` | the same protocol for SiLU, against Orion, ENSI, Sylph and THOR | Llama 3.2 3B Instruct |
| `SPA_FHE/` | ciphertext microbenchmarks under CKKS: latency, operation counts, consumed levels and a standardised error decomposition of every method on fixed intervals `[−5, 5]` and `[−20, 20]` | Microsoft SEAL 4.1 |

The plaintext harnesses select each method's parameters on calibration data
only, freeze them into a profile catalog, and evaluate zero-shot on held-out
splits. The ciphertext benchmark reads the same tier rule and, for MOAI, the
same profiles the plaintext model ran.

## Quick start

Each component has its own README with the full protocol, options and
outputs. The order that reproduces the paper is:

```bash
# 1. Plaintext, Gemma / GELU (GPU)
cd SPA_GELU
pip install -r requirements.txt
python tests/test_mcq_harness.py && python tests/test_profiles.py && python tests/test_paper_tables.py
python sanity_check_model.py --model-path <gemma-4-e2b-it>
MODEL_PATH=<gemma-4-e2b-it> bash run_all_tasks.sh                # tanh-GELU (the model's activation) as reference
MODEL_PATH=<gemma-4-e2b-it> TARGET=erf bash run_all_tasks.sh     # exact erf-GELU substituted as the common target
MODEL_PATH=<gemma-4-e2b-it> TARGET=erf bash run_rounding_experiment.sh   # coefficient-precision experiment

# 2. Plaintext, Llama / SiLU (GPU)
cd ../SPA_SiLU
MODEL_PATH=<llama-3.2-3b-instruct> bash run_all_tasks.sh
MODEL_PATH=<llama-3.2-3b-instruct> bash run_rounding_experiment.sh

# 3. Ciphertext (CPU, Microsoft SEAL 4.1 with a tc128 entry at N = 65536)
cd ../SPA_FHE
python3 scripts/import_model_moai_profiles.py \
    --catalog ../SPA_GELU/results/zeroshot_erf/hellaswag/profile_catalog.json \
    --cross-check ../SPA_GELU/results/zeroshot_erf/*/profile_catalog.json
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSEAL_DIR=<seal>/lib/cmake/SEAL-4.1
cmake --build build -j && ctest --test-dir build --output-on-failure
python3 scripts/validate_moai_model_profiles.py --host-check-binary build/bin/moai_host_check --plaintext-dir ../SPA_GELU
MODE=full REPEATS=10 WARMUPS=2 bash scripts/run_all.sh
```

Step 3 depends on step 1: the ciphertext MOAI is the plaintext model's own
profile, imported verbatim from the erf-GELU catalog.

## Layout of results

Plaintext runs write to `results/zeroshot/` (tanh-GELU reference) and
`results/zeroshot_erf/` (erf-GELU reference) inside `SPA_GELU`, and to
`results/zeroshot/` inside `SPA_SiLU`: per task, the calibration, the
frozen range plan, the profile catalog, per-example predictions and a
summary; at the top, pooled analyses and the paper tables
(`paper_main_*.csv`, `paper_main.md`). The ciphertext benchmark writes
`results/gelu_erf/`, `results/gelu_quickgelu/` and `results/silu/` with one
summary CSV per run (`r5`, `r20`, Experiment B `roundB_*`, Experiment C
`precC_*`), plus the error–latency figure. `results/`, `build/` and logs are
ignored by git.

## Requirements

* Plaintext: Python ≥ 3.10, a CUDA GPU with ≥ 48 GB (95 GB used for the
  reported runs), PyTorch ≥ 2.4, transformers ≥ 5 (Gemma 4), datasets,
  numpy, scipy. See each `requirements.txt`.
* Ciphertext: a C++17 compiler, CMake ≥ 3.16, Microsoft SEAL 4.1 built with
  a 128-bit security table entry at ring dimension 65536 (the benchmark
  refuses secure runs otherwise; `--no-sec` is diagnostic only), Python 3
  with numpy and scipy for the generators and checks, matplotlib for the
  figure.

## License

See `LICENSE`.

## Citation

```
@inproceedings{spa2026,
  title     = {SPA: Structure-Preserving Approximation of Gated Activations under FHE},
  year      = {2026}
}
```

