# StyleGAN2-ADA Gradient Explosion Fixes - Quick Reference

## What Was Fixed

### 1. R1 Regularization Stability ✅
**File:** `training/trainer.py` (Line 173)
- **Before:** `r1_penalty = r1_grads.pow(2).sum([1, 2, 3]).mean()`
- **After:** `r1_penalty = (r1_grads.pow(2).sum([1, 2, 3]) + 1e-6).mean()`
- **Why:** Prevents numerical underflow/overflow when gradients are tiny

### 2. AMP/FP32 Mixing Bug ✅
**File:** `training/trainer.py` (Lines 176-178)
- **Before:** Mixed FP32 R1 penalty directly with FP16 loss
- **After:** Convert R1 term to match d_loss dtype before addition
- **Why:** Prevents gradient scale mismatch causing NaNs

### 3. Learning Rate Reduction ✅
**File:** `configs/training_config.yaml` (Lines 23-24, 29)
- **Before:** G: 0.002, D: 0.002, r1_gamma: 10.0
- **After:** G: 0.0015, D: 0.001, r1_gamma: 5.0
- **Why:** Original rates were 2-4x too aggressive

### 4. Demodulation Epsilon ✅
**File:** `models/stylegan2_ada.py` (Line 53)
- **Before:** `+ 1e-8`
- **After:** `+ 1e-5`
- **Why:** Better numerical stability in FP16

### 5. Gradient Monitoring ✅
**File:** `training/trainer.py` (Lines 184-189, 218-223)
- Added NaN detection before backward pass
- Added gradient explosion monitoring (> 100.0)
- Automatic training halt on NaN

### 6. Configurable Paths ✅
**File:** `training/trainer.py` (Lines 62, 65)
- **Before:** Hardcoded `/content/drive/MyDrive/`
- **After:** Config-based with fallback to `./checkpoints`
- **Why:** Support both local and cloud training

## How to Use

### Quick Test (150 iterations)
```bash
python test_gradient_fixes.py
```

### Local Training (500 iterations)
```bash
python train.py --config configs/local_config.yaml
```

### Full Training (after local validation)
```bash
python train.py --config configs/training_config.yaml
```

## Expected Results

After fixes, training should show:
- ✅ D loss: 0.5 - 2.0 (NOT collapsing to 0.000)
- ✅ G loss: 2.0 - 15.0 (NOT spiking to 100+)
- ✅ Gradient norms: < 10.0
- ✅ No NaN errors
- ✅ Passes iteration 100 smoothly

## Files Modified

1. `training/trainer.py` - Core training fixes
2. `models/stylegan2_ada.py` - Demodulation epsilon
3. `configs/training_config.yaml` - Updated hyperparameters
4. `configs/local_config.yaml` - Local testing config (NEW)
5. `test_gradient_fixes.py` - Validation script (NEW)

## Dataset Configuration

Your dataset path is correctly set to:
```
F:\stylegan2-face-synthesis\archive\img_align_celeba\img_align_celeba
```

Checkpoints will save to:
```
F:\stylegan2-face-synthesis\checkpoints
```

Samples will save to:
```
F:\stylegan2-face-synthesis\outputs\samples
```
