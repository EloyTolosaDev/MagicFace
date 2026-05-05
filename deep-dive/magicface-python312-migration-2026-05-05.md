# Deep Dive: MagicFace Python 3.12 Dependency Migration

**Generated**: 2026-05-05 17:56:48 CEST  
**Phase**: Python 3.12 migration and dependency verification  
**Files**: `pyproject.toml`, `.gitignore`, `README.md`, `utils/preprocess.py`, `utils/retrieve_bg.py`, removed `requirements.txt`, removed tracked `__pycache__/*.pyc`

---

## Overview

### What This Code Does

This change migrates MagicFace away from a frozen `requirements.txt` and into a Python 3.12 installable project using `pyproject.toml`. It also removes interpreter-specific bytecode files that were committed from Python 3.8 and Python 3.10, and makes two preprocessing code paths safer with current dependency behavior.

The result is a smaller, resolver-friendly dependency surface:

- Direct dependencies live in `pyproject.toml`.
- PyTorch owns its own CUDA package pins.
- InsightFace preprocessing checks ONNX Runtime providers directly instead of assuming ONNX CUDA support from `torch.cuda.is_available()`.
- Pillow resizing uses the modern `Image.Resampling.BILINEAR` enum.

### Why This Approach Was Chosen

The old `requirements.txt` was a transitive freeze. It pinned direct libraries, indirect libraries, CUDA runtime wheels, and packages that should normally be chosen by pip's resolver. That makes upgrades brittle: one old or mismatched transitive pin can force pip into backtracking or a conflict.

The migration keeps direct pins only where compatibility is important. The most important compatibility decision is the Hugging Face stack:

- The custom MagicFace pipeline imports internal Diffusers modules that match the Diffusers 0.25 generation.
- `diffusers==0.25.1`, `huggingface-hub==0.25.2`, and `transformers==4.48.3` were verified together in Python 3.12.
- Newer Diffusers would require a larger port because internal modules such as LoRA compatibility layers and pipeline helper APIs have moved over time.

This is intentionally surgical: it stabilizes Python 3.12 without rewriting the model pipeline.

### Context

Use this pattern when a research repo has custom code copied from a framework internals layer. The safest migration is not always "latest of everything." It is often a compatibility island: keep the framework version close to the copied code, then modernize only the environment and small runtime assumptions around it.

---

## Code Walkthrough

### `pyproject.toml`

**Purpose**: Replaces `requirements.txt` with project metadata and direct dependencies.

**Key Components**:

- `[build-system]`: Uses setuptools as the build backend, so `pip install -e .` works.
- `requires-python = ">=3.12,<3.13"`: Makes Python 3.12 support explicit and prevents accidental installs under older interpreters.
- Critical pins:
  - `torch==2.3.1` and `torchvision==0.18.1`: Matching pair for the existing model code.
  - `diffusers==0.25.1`, `huggingface-hub==0.25.2`, `transformers==4.48.3`: Compatible Hugging Face stack verified by pip and import smoke tests.
  - `numpy==1.26.4` and `opencv-python-headless==4.11.0.86`: Avoids pulling OpenCV builds that expect NumPy 2 while this stack is still on NumPy 1.26.
  - `onnxruntime>=1.20,<2`: Provides the CPU ONNX Runtime backend needed by InsightFace.
- `[tool.setuptools.packages.find]`: Includes both `mgface*` and `utils*`, including namespace packages.

**Line-by-Line Breakdown**:

```toml
# Lines 1-3: pip can build/editably install the repo through setuptools.
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

# Lines 5-10: package metadata and Python 3.12-only support boundary.
[project]
requires-python = ">=3.12,<3.13"

# Lines 11-28: direct dependency contract, not a transitive freeze.
dependencies = [
    "torch==2.3.1",
    "torchvision==0.18.1",
    "diffusers==0.25.1",
]
```

### `.gitignore`

**Purpose**: Prevents new bytecode, local venvs, and editable-install metadata from re-entering the repository.

**Why It Matters**: Python bytecode is interpreter-specific. The removed tracked files included names such as `cpython-310.pyc` and `cpython-38.pyc`, which are not source and should not be part of a Python 3.12 migration.

### `README.md`

**Purpose**: Updates installation instructions from `pip install -r requirements.txt` to `pip install -e .`.

**Key Change**:

```console
python -m pip install -e .
```

The README now documents that dependencies are declared in `pyproject.toml`, and it calls out the default `onnxruntime` CPU behavior for preprocessing.

### `utils/preprocess.py`

**Purpose**: Initializes InsightFace for image cropping.

**Key Components**:

- `os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")`: Prevents Albumentations' import-time update check from trying to reach the network through InsightFace's dependency tree.
- `get_onnx_providers()`: Reads available ONNX Runtime providers from ONNX Runtime itself.
- `providers, ctx_id = get_onnx_providers()`: Uses ONNX Runtime capability, not PyTorch CUDA capability, to configure InsightFace.

**Why It Changed**:

The old code used `torch.cuda.is_available()` to decide whether to pass `CUDAExecutionProvider` to InsightFace. That is not the same as ONNX Runtime having a CUDA provider installed. With CPU `onnxruntime`, PyTorch can be CUDA-capable while ONNX Runtime is not.

The new code asks ONNX Runtime directly:

```python
def get_onnx_providers():
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1
```

### `utils/retrieve_bg.py`

**Purpose**: Builds MagicFace background and pose conditioning inputs.

**Key Components**:

- Same ONNX Runtime provider detection as `preprocess.py`.
- `RESAMPLE_BILINEAR = Image.Resampling.BILINEAR`: Uses Pillow's current resampling enum.
- `image = img.resize((512, 512), RESAMPLE_BILINEAR)`: Replaces the legacy `Image.BILINEAR` constant.

**Why It Changed**:

Pillow has moved image resampling constants under `Image.Resampling`. Using the enum keeps this path compatible with modern Pillow while still expressing the same resizing behavior.

---

## Concepts Explained

### Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| Direct dependency manifest | `pyproject.toml` | Keeps only project-owned requirements in source control and lets pip resolve transitive packages. |
| Compatibility island | Hugging Face pins in `pyproject.toml` | Preserves old Diffusers internals used by MagicFace without a large framework port. |
| Capability discovery | `get_onnx_providers()` | Uses the runtime's own provider list instead of inferring ONNX support from PyTorch. |
| Import-time side-effect guard | `NO_ALBUMENTATIONS_UPDATE` before importing InsightFace | Prevents a transitive dependency from doing network checks during import. |

### Key Technical Concepts

#### Direct vs. Transitive Dependencies

**What**: A direct dependency is something this project imports or intentionally relies on. A transitive dependency is pulled in by another package.

**Why Used Here**: The old file pinned both, which made the environment fragile. The new manifest declares direct dependencies and lets pip choose compatible transitive versions.

**When to Use**: Prefer direct manifests for application repos unless you are creating a lock file with a tool designed for locking.

**Trade-offs**:

- Pros: Less resolver conflict, clearer ownership, smaller diffs during upgrades.
- Cons: Installs can drift unless critical direct versions are pinned.

**Alternatives**:

- Keep `requirements.txt`: simple, but repeats the original problem.
- Use a lock file tool: good for production reproducibility, but heavier than needed for this repo's current request.

#### Compatibility Island

**What**: A deliberately bounded set of versions kept together because code depends on old framework internals.

**Why Used Here**: MagicFace imports Diffusers internals in `mgface/pipelines_mgface/*`. Upgrading Diffusers broadly would likely require editing many copied pipeline/model files.

**When to Use**: Use this when custom research code vendors or subclasses non-public framework APIs.

**Trade-offs**:

- Pros: Small, testable migration.
- Cons: You inherit warnings from older framework versions, such as Diffusers using deprecated PyTorch pytree registration.

#### Provider Discovery

**What**: Runtime capability is checked from the library that will execute the work.

**Why Used Here**: ONNX Runtime providers are independent from PyTorch CUDA support. The correct question is "does ONNX Runtime expose `CUDAExecutionProvider`?", not "does PyTorch see CUDA?"

**When to Use**: Any time two different runtimes can use different hardware backends.

---

## Verification

### Commands Run

```console
venv/bin/python -m pip install -e .
venv/bin/python -m pip check
PYTHONPYCACHEPREFIX=/tmp/magicface-pycache venv/bin/python -m compileall -q inference.py mgface utils
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/magicface-mpl venv/bin/python -c "<import smoke test>"
PYTHONDONTWRITEBYTECODE=1 venv/bin/python inference.py --help
```

### Results

- Editable install completed successfully in Python 3.12.3.
- `pip check` reported: `No broken requirements found.`
- Python 3.12 compilation succeeded when writing bytecode to `/tmp/magicface-pycache`.
- Import smoke test loaded:
  - `torch 2.3.1+cu121`
  - `torchvision 0.18.1+cu121`
  - `diffusers 0.25.1`
  - `transformers 4.48.3`
  - `huggingface_hub 0.25.2`
  - `numpy 1.26.4`
  - `opencv 4.11.0`
  - `onnxruntime 1.25.1`
- `inference.py --help` imported the custom MagicFace pipeline and printed CLI help successfully.

### Known Local Notes

- Full image generation was not run because the required model files are not present locally and the inference script would need to download large Hugging Face assets.
- A pre-existing untracked root `__pycache__/inference.cpython-312.pyc` is owned by `nobody`, so normal `compileall` cannot write there. The source tree now ignores `__pycache__`, and compilation was verified with `PYTHONPYCACHEPREFIX=/tmp/magicface-pycache`.
- ONNX Runtime printed a device discovery warning in this sandbox, but available providers were still detected as `['AzureExecutionProvider', 'CPUExecutionProvider']`, and the code correctly selected CPU.

---

## Learning Resources

### Official Documentation

- [PyPA: Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/): Explains `[build-system]`, `[project]`, `dependencies`, and `requires-python`.
- [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/): Shows the official `torch==2.3.1` / `torchvision==0.18.1` pairing.
- [Diffusers 0.25.1 package metadata](https://pypi.org/pypi/diffusers/0.25.1/json): Useful for checking Python and dependency metadata for the pinned Diffusers version.
- [Transformers 4.48.3 package metadata](https://pypi.org/pypi/transformers/4.48.3/json): Useful for checking the Transformers version selected for the Hugging Face compatibility set.
- [Hugging Face Hub 0.25.2 package metadata](https://pypi.org/pypi/huggingface-hub/0.25.2/json): Confirms the Hub client version paired with the old Diffusers stack.
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/): Explains provider priority such as `['CUDAExecutionProvider', 'CPUExecutionProvider']`.

### Related Concepts For Deeper Study

- Dependency locking vs. dependency declaration.
- Python namespace packages and editable installs.
- CUDA runtime packages as PyTorch transitive dependencies.
- Risks of depending on framework internals in research repositories.

---

## Related Code in This Project

| File | Relationship |
|------|--------------|
| `inference.py` | Imports the custom MagicFace pipeline and the pinned Diffusers/Transformers APIs. |
| `mgface/pipelines_mgface/*.py` | Custom Diffusers-era pipeline and UNet code that motivated keeping `diffusers==0.25.1`. |
| `utils/preprocess.py` | Uses InsightFace and ONNX Runtime for cropping source images. |
| `utils/retrieve_bg.py` | Uses InsightFace, ONNX Runtime, Pillow, and PyTorch to produce background/pose conditioning. |
| `README.md` | Documents the new Python 3.12 installation flow. |

---

## Next Steps

1. **Try it yourself**: After downloading model assets, run the README inference command in the Python 3.12 venv.
2. **Deeper dive**: If you later want newer Diffusers, first map every `diffusers.*` internal import in `mgface/pipelines_mgface` to its modern equivalent.
3. **Common pitfalls**: Do not re-freeze the full environment into `requirements.txt`; if a lock file is needed, use a dedicated lock tool and keep it separate from direct dependency metadata.

---

*This deep dive was generated by AntiVibe - the anti-vibecoding learning framework.*  
*Learn what AI writes, not just accept it.*
