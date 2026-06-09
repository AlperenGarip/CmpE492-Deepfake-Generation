# Puppet & Painter — Hybrid Audio-Driven Talking Head Generation

CMPE 492 Graduation Project · Boğaziçi University · Alperen Garip · Advisor: Berk Gökberk

A hybrid pipeline that decouples **motion generation** from **rendering**: a 3D-aware "Puppet" (MimicTalk) generates audio-synchronized motion, and a deterministic warping "Painter" (LivePortrait) re-renders it photorealistically.

**Key result (10 HDTF identities, cross-audio protocol):** the hybrid improves lip-sync **7.4×** over the renderer-only baseline (LSE-C 5.68 vs 0.76, p=0.002) while preserving identity significantly better than Wav2Lip (CSIM 0.941 vs 0.825, p=0.002), at no visual-quality cost.

## Repository Structure

### Phase 1–3: Pipeline Development (single-identity)

| Notebook | Purpose |
|---|---|
| `1_MimicTalk_CMPE492.ipynb` | Phase 1 — MimicTalk (the Puppet): personalization + audio-driven inference |
| `2_AniPortrait_CMPE492.ipynb` | Phase 2 — AniPortrait diffusion baseline |
| `3A_Phase3_Hybrid_AniPortrait.ipynb` | Phase 3A — Hybrid with diffusion Painter (negative result: diffusion destroys lip-sync) |
| `3B_Phase3_Hybrid_LivePortrait.ipynb` | Phase 3B — **Hybrid-LP, the main method** (MimicTalk → LivePortrait) |

### Multi-Identity Benchmark (main evaluation)

Run in order; each stage's output zip feeds the next.

| Notebook | Purpose |
|---|---|
| `4_HDTF_DataPrep.ipynb` | Downloads & standardizes 10 HDTF identities (512×512, 25 fps) → `hdtf_prepared.zip` |
| `5_MimicTalk_Batch_CrossAudio.ipynb` | Personalizes & runs MimicTalk per identity with **cross-identity audio** (cyclic id→id+1 mapping) → `mimic_videos.zip` |
| `6_LivePortrait_Batch_CrossAudio.ipynb` | Renders Hybrid-LP and the LP-only controlled baseline for all identities → `batch_eval.zip` |
| `7_Wav2Lip_Baseline.ipynb` | External SOTA baseline (Wav2Lip) under the same cross-audio protocol → `wav2lip_videos.zip` |
| `8_Extended_Metrics.ipynb` | FID, FVD, LPIPS, PSNR, SSIM |
| `9_Final_Evaluation.ipynb` | SyncNet (LSE-C/D, AV offset) + ArcFace (CSIM) scoring for all 3 configs + paired Wilcoxon significance tests → `results/` |

### Results

| File | Contents |
|---|---|
| `results/final_results.json` | Per-identity metrics, aggregates (mean±std), significance tests |
| `results/final_per_identity.csv` | Flat per-identity table (config × identity × metrics) |
| `MidReport.pdf` | Full project report |

## Why cross-identity audio?

If each identity were driven by its **own** audio, the LP-only baseline (which re-renders the original video) would trivially retain perfect lip-sync, masking the Puppet's contribution. Driving each identity with a *different* identity's audio ensures the baseline's lips do **not** match the audio — isolating the effect of audio-driven motion generation.

## Main Results

| Method | LSE-C ↑ | LSE-D ↓ | AV offset | CSIM ↑ |
|---|---|---|---|---|
| LP-only (baseline) | 0.76 ± 0.18 | 14.76 ± 0.41 | −5.9 ± 9.9 | 0.934 ± 0.024 |
| Wav2Lip (SOTA) | **7.26 ± 0.74** | 14.66 ± 0.22 | −2.0 ± 0.0 | 0.825 ± 0.038 |
| **Hybrid-LP (ours)** | 5.68 ± 1.26 | **14.34 ± 0.26** | **0.7 ± 0.5** | **0.941 ± 0.018** |

Paired Wilcoxon (n=10): Hybrid vs LP-only LSE-C **p=0.002**; Hybrid vs Wav2Lip CSIM **p=0.002**.

## Environment

All notebooks target **Google Colab with a GPU runtime**. Environment patches (NumPy/SciPy pinning for SyncNet, `pkg_resources` fixes for Python 3.12, HuBERT download workarounds) are included in each notebook's setup cells. A **runtime restart** is required wherever a notebook says so (after NumPy reinstall / MimicTalk install).

## References

MimicTalk (Lu et al., NeurIPS 2024) · LivePortrait (Guo et al., 2024) · AniPortrait (Wei et al., 2024) · Wav2Lip (Prajwal et al., ACM MM 2020) · SyncNet (Chung & Zisserman) · ArcFace (Deng et al.) · HDTF (Zhang et al., CVPR 2021)
