# Wake Word Notebook — Fixes Applied

Notes on errors hit while running the wake-word training notebook (2026-08), and the fixes applied. Base notebook: alfiedennen/openwakeword-colab-2026. Patched here: impostor-dax/train_openwake_fix.

## 1. Missing `deep-phonemizer` dependency

**Error:**
```
ModuleNotFoundError: No module named 'dp'
```
Hit during the `generate_clips` step, which generates adversarial negative text samples using DeepPhonemizer.

**Fix:** install the missing package before running `generate_clips`:
```python
!pip install deep-phonemizer
```

## 2. PyTorch 2.6 `weights_only` default change

**Error:**
```
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL dp.preprocessing.text.Preprocessor was not an allowed global by default.
```
As of PyTorch 2.6, `torch.load` defaults to `weights_only=True` for security. DeepPhonemizer's checkpoint loader doesn't set this flag, so it fails.

**First attempt (didn't work):** monkey-patching `torch.load` in the notebook's own Python session. This has no effect because `train.py` runs as a separate subprocess (`!{sys.executable} ... train.py`), so the patch doesn't carry over.

**Working fix:** patch the installed DeepPhonemizer package file directly, so it applies regardless of which process loads it:
```python
path = '/usr/local/lib/python3.12/dist-packages/dp/model/model.py'
with open(path) as f:
    content = f.read()
content = content.replace(
    "checkpoint = torch.load(checkpoint_path, map_location=device)",
    "checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)"
)
with open(path, 'w') as f:
    f.write(content)
```

Run this once, before (re-)running the `generate_clips` cell.

## Order of operations

If setting up fresh:
1. Run the notebook as normal through the install/setup cells.
2. Before the `generate_clips` cell, insert and run the two fix cells above (deep-phonemizer install, then the model.py patch).
3. Re-run `generate_clips` — should complete without the earlier errors.
4. Continue with the rest of the notebook as documented upstream.

## Notes
- Both issues are environment/dependency drift in DeepPhonemizer + PyTorch, not bugs in the openwakeword-colab-2026 notebook itself.
- Not yet reported upstream to alfiedennen/openwakeword-colab-2026 — worth opening an issue there since this will hit anyone running the notebook on current Colab.
