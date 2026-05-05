# MagicFace
Official implementation of MagicFace

> **MagicFace: High-Fidelity Facial Expression Editing with Action-Unit Control** [[arXiv paper](http://arxiv.org/abs/2501.02260)]<br>
> Mengting Wei, Tuomas Varanka, Xingxun Jiang, Huai-Qian Khor, Guoying Zhao<br>
> University of Oulu


## Introduction
We address the problem of facial expression editing
by controling the relative variation of facial action-unit (AU) from
the same person. This enables us to edit this specific person’s expression in a fine-grained, continuous and interpretable manner,
while preserving their identity, pose, background and detailed
facial attributes. By injecting AU variations
into a denoising UNet, our model can animate arbitrary identities
with various AU combinations, yielding superior results in high-fidelity expression editing compared to other facial expression
editing works.


<p align="center"> 
<img src="./assets/demo.jpg">
</p>



### Dependencies

- Python 3.12 (tested with Python 3.12.3)
- Your computer should have a graphics card with approximately **8GB** to support running this test.

### Installation

Create a new Python 3.12 environment and install the dependencies:

```console
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml`. The core pins keep the custom Diffusers pipeline on the compatible `diffusers==0.25.1`, `huggingface-hub==0.25.2`, and `transformers==4.48.3` set while using Python 3.12.

The default install uses `onnxruntime` for CPU preprocessing. If you need ONNX CUDA execution for `preprocess.py` or `retrieve_bg.py`, install the matching `onnxruntime-gpu` build for your CUDA setup after the core install.

Runtime model caches are stored under the repository-local `.models/` directory:

- `.models/huggingface`: Hugging Face model cache used by `inference.py`.
- `.models/insightface`: InsightFace model root. InsightFace looks for `models/antelopev2` inside this directory.

### Download Models

The scripts download required models into `.models/` automatically on first use. To pre-download them explicitly:

```console
python -m utils.model_assets
```

This fetches MagicFace assets from [HuggingFace](https://huggingface.co/mengtingwei/magicface/tree/main) into `.models/magicface`, keeps the Hugging Face cache under `.models/huggingface`, and fetches InsightFace `antelopev2` into `.models/insightface`.


### Usage

#### Using our examples

You can test our model by editing the images we provided. Model inference needs an identity image
to edit, a background image for attribute condition and an AU condition. 

> This script will directly download the model by the model cards of Huggingface, so the first time testing the download may take a lot of time. 

* Test the model:

```--au_test``` The AUs you want to modify for the face. We provide 12 editable AUs here.
They are _AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26_. Only provide the AUs 
you want to modify here and split them by ``+``. For example, the following example shows how to 
edit AU1 and AU4. If only one AU is intended to change, just provide that one.

```--AU_variation```  Intensity integers you want to edit for each AU you specified. Also split them by ``+`` 
if multiple AUs are intended to change. We recommend to limit the intensity 
in the range of [-10, 10]. Integers outside this range may experience severe distortion.



```console
python inference.py --img_path './test_images/00381.png' --bg_path './test_images/00381_bg.png' --au_test 'AU4+AU1' --AU_variation '4+2'
```



#### Test your own images

If you want to edit your own images, you need to compute the 
background and pose for attribute condition.

1. Check that model assets are available locally. They are downloaded automatically, but after download the local structure should be like this:

```
.models
    magicface
        utils
            79999_iter.pth
            third_party
                ...
            checkpoints
                ...
    insightface
        models
            antelopev2
                ...
```

2. Crop your image into the resolution of 512 $\times$ 512. Please provide
the image including at least one face, otherwise it will result in an error.
```console
cd utils
python preprocess.py --img_path <your-image-path> --save_path <your-save-path>
```

3. Then parse the background and draw the contour from the cropped image.

```console
python retrieve_bg.py --img_path <your-cropped-path> --save_path <your-save-path>
```
4. Use the `inference.py` script introduced above to test your image.
### Issues or Questions?
If the issue is code-related, please open an issue here.

For questions, please also consider opening an issue as it may benefit future reader. 
Otherwise, email Mengting Wei at [mengting.wei@oulu.fi](mengting.wei@oulu.fi).

### Acknowledgements

This codebase was built upon and drew inspirations from [FineFace](https://github.com/tvaranka/fineface), [InsightFace](https://github.com/deepinsight/insightface),
[BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) and [Stable DIffusion](https://github.com/CompVis/stable-diffusion). 

We thank the authors for making those repositories public.
