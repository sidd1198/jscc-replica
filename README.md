# Small targets fail first: a learning replication of scale-aware Deep JSCC

A from-scratch, learn-by-doing replication of the core experiment in
*"Small Targets Fail First: Detector-Agnostic Scale-Aware Deep JSCC for Aerial ISR Downlinks"*
(Yashwanth M et al.). This is **not** the authors' code and not a full reproduction; it is a smaller,
cheaper version built to learn the method end to end. Numbers here come from this repo's own runs and
are not expected to match the paper exactly (see *Differences from the paper*).

_Last updated: 2026-11-13_

## The idea in four lines
1. A drone sends images over a noisy radio link to a ground station that runs an object detector.
2. Instead of JPEG + error-correcting code, a small neural codec (encoder, noisy channel, decoder) is
   trained end to end, so quality degrades gradually instead of falling off a cliff.
3. The detector (YOLOv8n) is **frozen**: it receives no gradient and is only used to score the output.
4. Detection is scored as AP **per object size** (tiny < 256 px², small 256-1024 px², large >= 1024 px²),
   and the codec is trained with an inverse-area weight map that makes small-object pixels count more.

## What was built
| File | Purpose |
|---|---|
| `data.py` | Parse original VisDrone-DET annotations, print box-size statistics (fails loudly if no boxes are found) |
| `evaluate.py` | AP@0.5 by object size for clean images (the "ceiling") |
| `channel.py` | Power normalisation + AWGN / Rician / Rayleigh channel (differentiable), with self-tests |
| `model.py` | SNR-adaptive encoder/decoder (about 0.66M parameters each at c = 24) |
| `loss.py` | Inverse-area weight map and weighted MSE (lambda = 0 gives plain MSE exactly) |
| `train.py` | Training loop (random 384 px crops, SNR drawn per batch, resumes from `last.pt`) |
| `eval_codec.py` | image -> codec -> channel -> frozen YOLO -> AP by size at several SNRs |
| `compare.py` | Small/tiny AP of two result files side by side |
| `sweep_table.py` | AP by size for every lambda run, with change vs lambda = 0 |
| `jpeg_baseline.py` | JPEG + capacity-achieving-code baseline (written; see *Not completed*) |
| `plot_vs_jpeg.py` | Small-object AP against SNR for several result files |
| `overfit.py` | 8-image sanity check that the model can learn at all |
| `weights/best.pt` | The frozen detector (about 6 MB, tracked in git) |
| `results/` | JSON/PNG outputs of every evaluation |

Not tracked by git (see `.gitignore`): `data/`, `runs/`, `models/`, other `*.pt` files, `.venv/`.

## Setup used
- Data: VisDrone2019-DET, 6,471 train / 548 val images. Val: 38,759 boxes, median area 520 px²
  (paper: 533), size bins 30.8% / 37.7% / 31.4% (matches the paper).
- Detector: YOLOv8n fine-tuned **5 epochs at 640 px** on Colab (Ultralytics val mAP@0.5 = 0.233).
- Clean-image ceiling with this repo's own scorer (AP@0.5, 640 px): tiny 0.0439, small 0.1483,
  large 0.3430, all 0.2008.
- Codec: c = 24 (bandwidth ratio R = 1/16), **15 epochs**, batch 16, Adam 1e-4 with cosine decay,
  gradient clipping 1.0, SNR drawn uniformly from 0-20 dB per batch, trained on AWGN only.
- Seeds: 0 and 1. Training speed: about 106 s/epoch on a Colab T4, about 73 s/epoch on an RTX 4050 laptop.

## Results (AWGN, evaluated on all 548 val images)

### 1. Small objects fail first (lambda = 0, share of clean-image AP retained)
| SNR | tiny | small | large |
|---|---|---|---|
| 0 dB, seed 0 | 35.9% | 49.7% | 53.1% |
| 0 dB, seed 1 | 35.6% | 48.5% | 51.0% |
| 20 dB, seed 0 | 74.6% | 82.5% | 87.3% |
| 20 dB, seed 1 | 79.9% | 84.5% | 88.4% |

The ordering tiny < small < large holds at every SNR in both seeds. The gap between sizes is smaller
than in the paper (see *Differences*), probably because the weak 640 px detector leaves tiny objects with
a very low ceiling.

### 2. Weighted loss (lambda = 32) against plain MSE: small-object AP
| SNR | lambda 0 (s0 / s1) | lambda 32 (s0 / s1) | change (s0 / s1) | two-seed mean change |
|---|---|---|---|---|
| 0 dB | 0.0737 / 0.0718 | 0.0722 / 0.0731 | -2.0% / +1.8% | about 0% |
| 5 dB | 0.1037 / 0.1045 | 0.1110 / 0.1130 | +7.0% / +8.1% | +7.6% |
| 10 dB | 0.1176 / 0.1197 | 0.1255 / 0.1260 | +6.7% / +5.2% | +6.0% |
| 15 dB | 0.1220 / 0.1210 | 0.1307 / 0.1318 | +7.1% / +8.9% | +8.0% |
| 20 dB | 0.1223 / 0.1253 | 0.1325 / 0.1350 | +8.4% / +7.7% | +8.0% |

- From 5 dB upward the gain is 0.006-0.011 AP in both seeds, above the observed baseline seed spread
  (about 0.002-0.003 AP). The paper reports +8.2% at 20 dB; this repo gets +8.0% (two-seed mean).
- The gain is **not specific to small objects**: large-object AP also rises about 4-6% at 10-20 dB in both
  seeds, and overall AP by 3-8%.
- Tiny-object gains are positive at 10-20 dB in both seeds but not reliable in size (+33% in seed 0, +9% in
  seed 1 at 20 dB) and slightly negative at 0-5 dB in seed 1. Treat them as inconclusive.
- The paper's 0 dB gain (+4.5%) is **not reproduced**: here it is about zero and changes sign between seeds.

### 3. Rician fading (K = 10 dB), same seed-0 models, no retraining
Small-object AP change from the weighted loss: +2.3% (0 dB), +5.9% (5), +9.0% (10), +7.4% (15), +6.6% (20).
For lambda = 0, 20 dB small AP is 0.1225 on Rician against 0.1223 on AWGN (matching the paper's finding
that Rician is within the noise floor at high SNR); at 0 dB it is 0.0666 against 0.0737.

## Reproduce (Windows PowerShell)
```powershell
# one-time
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install numpy matplotlib pillow ultralytics   # then install the CUDA build of PyTorch from pytorch.org

# data: use the ORIGINAL VisDrone zips, not the converted copy Ultralytics creates
# https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip (and -val.zip)
# unzip into data\  ->  data\VisDrone2019-DET-train\{images,annotations}, same for val
python data.py                                    # expect val: 548 images, 38759 boxes

# detector (Colab T4):  yolo detect train model=yolov8n.pt data=VisDrone.yaml epochs=5 imgsz=640 batch=16
# put the resulting best.pt in weights\
python evaluate.py                                # clean ceiling by size

# codec
python train.py --lam 0  --epochs 15 --seed 0 --out runs\codec_lam0  --num-workers 2
python train.py --lam 32 --epochs 15 --seed 0 --out runs\codec_lam32 --num-workers 2
python eval_codec.py --ckpt runs\codec_lam0\final.pt  --out results\lam0.json
python eval_codec.py --ckpt runs\codec_lam32\final.pt --out results\lam32.json
python compare.py results\lam0.json results\lam32.json
```
Use `--channel rician` or `--channel rayleigh` in `eval_codec.py` to change the test channel, and
`--seed 1` plus a new `--out` folder for another seed.

## Differences from the paper
- Detector trained 5 epochs at 640 px (paper: 40 epochs at 960 px, mAP@0.5 = 0.389). All absolute APs are lower.
- Codec trained 15 epochs (paper: 40); 2 seeds (paper: 3, with a t-test).
- Own AP scorer (COCO-style handling of unmatched detections, 101-point AP at IoU 0.5); its overall AP (0.2008) is
  lower than Ultralytics' validation mAP@0.5 (0.233) for the same detector.
- Median val box area is 520 px² against 533 in the paper (probably a small filtering difference).

## Lessons learned while building it
- Ultralytics' `VisDrone.yaml` converts labels to YOLO format and moves the images; code written for the original
  VisDrone format then silently sees **zero** boxes (every AP is 0 and a weighted loss quietly equals plain MSE).
  `data.py` now raises an error if no boxes are parsed.
- Test each stage before building on it: data statistics, scorer on clean images, channel with random numbers,
  model shapes, 8-image overfit, then training.
- Keep checkpoints off the notebook's temporary disk (Colab sessions wipe it) and use relative output paths.
- On Windows the default `pip install` of PyTorch is CPU-only; install the CUDA build explicitly.

## Not completed
Ordered by importance.
1. **JPEG + capacity-code baseline.** `jpeg_baseline.py` and `plot_vs_jpeg.py` are written and were tested on real
   VisDrone images for the bit-budget and quality search, but a full run with the real detector is not recorded
   here. The paper finds the baseline ahead of JSCC above about 5 dB (and far behind at 0 dB), so the neural
   codec's advantage should not be claimed at high SNR until this is run.
2. **Lambda sweep (8, 16, 64).** Trained and evaluated, but the numbers are not summarised here. Run
   `python sweep_table.py 0 5 10 15 20` and add the table to this file.
3. **gamma = 0 ablation** (`--lam 32 --gamma 0`): does the benefit come from weighting object pixels at all, or from
   favouring small objects? The large-object gains above make this the main open question.
4. **Third seed and a significance test** (the paper uses 3 seeds and a two-sample t-test).
5. **Rayleigh channel evaluation** (`channel.py` supports it; no results yet), and training on fading channels
   (all codecs here were trained on AWGN only).
6. **Other bandwidth ratios** (`--c 12` and `--c 48`, R = 1/32 and 1/8).
7. **Full-strength detector** (40 epochs at 960 px) and re-measuring every ceiling.
8. **Open questions:** why the 0 dB gain is absent, and how reliable the tiny-object gain is.