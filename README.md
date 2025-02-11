# Latent Compression Learning (LCL)

![Static Badge](https://img.shields.io/badge/NeurIPS-2024-red)
[![Static Badge](https://img.shields.io/badge/arXiv-2406.07543-green)](https://arxiv.org/abs/2406.07543)

**[NeurIPS 2024]** [**Vision Model Pre-training on Interleaved Image-Text Data via Latent Compression Learning**](https://arxiv.org/abs/2406.07543)

We introduce the Latent Compression Learning (LCL) to pre-train vision models from scratch with interleaved image-text data. Compared to existing methods (e.g., CLIP, auto-regressive text generation), our proposed LCL is the first to achieve both

* Learning vision models from scratch
* Training on interleaved image-text data

![overview](./assets/overview.png)

## 📈 Results

### Pre-training on MMC4 Dataset

![result_interleaved](./assets/result_interleaved.png)

Our LCL pre-training significantly outperforms all other methods in the caption tasks and is on par with the best paired pre-training methods on classification and retrieval tasks.

### Comparison with OpenCLIP

![result_main_transfer](./assets/result_main_transfer.png)

![result_main_multimodal](./assets/result_main_multimodal.png)

When both using LAION-400M data, our LCL pre-training achieves similar performance to OpenCLIP. When combined with MMC4 data, our LCL pre-training outperforms OpenCLIP, especially in caption and multi-modal dialogue tasks. For a fair comparison, the total number of images seen during pre-training is 13B.

## 📦 Pre-trained Checkpoints

| model | data | # samples | download |
| :---: | :---: | :---: | :---: |
| ViT-B/16 | LAION-400M | 13B | [config](./src/open_clip/model_configs/LCL_ViT-B-16_laion.json) / [ckpt](https://huggingface.co/OpenGVLab/LCL-ViT-B-16-Laion) |

## 🛠️ Usage

### Install

This code is built upon [OpenCLIP](https://github.com/mlfoundations/open_clip), you can refer to their repository for setup.

### Load Pre-trained Checkpoints

Here is an example code to load pre-trained checkpoints:

```python
import open_clip

model_name = "LCL_ViT-B-16_laion"
pretrained = "path to the `.pt` file"

model = open_clip.create_model(model_name, pretrained=pretrained)
```

### Train LCL

The example training scripts are provided in [`./scripts`](./scripts). You can refer to [OpenCLIP](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#training-clip) for more ways to launch training.

**Training on LAION-400M.** Here is an example training script: [`./scripts/lcl_vit_b_32_laion.sh`](./scripts/lcl_vit_b_32_laion.sh). The corresponding model config is [here](./src/open_clip/model_configs/LCL_ViT-B-32_laion.json).

**Training on MMC4.** We provide a simple dataloader that supports the original [MMC4](https://github.com/allenai/mmc4) dataset. Organize the data folder as follows:

```
  /path/to/mmc4/
      ├── images/
      │   └── ...
      └── data/ 
          ├── docs_shard_0_v2.jsonl.zip
          ├── docs_shard_1_v2.jsonl.zip
          └── ...
```

Here is an example training script: [`./scripts/lcl_vit_b_32_mmc4.sh`](./scripts/lcl_vit_b_32_mmc4.sh). The corresponding model config is [here](./src/open_clip/model_configs/LCL_ViT-B-32_mmc4.json).

More training scripts can be found under [`./scripts`](./scripts).

**NOTE:** We conduct large-scale pre-training with internal efficient code, which will not be released due to intellectual property reasons. This released version has been verified and can reproduce the results of ViT-B/16 on LAION-400M dataset.


## 📅 Schedule

* [X]  basic code of LCL
* [ ]  checkpoints of more models and datasets
* [ ]  transfer evaluation code

## 🖊️ Citation

If you find this work helpful in your research, please consider citing:

```bibtex
@article{yang2024vision,
  title={Vision Model Pre-training on Interleaved Image-Text Data via Latent Compression Learning},
  author={Yang, Chenyu and Zhu, Xizhou and Zhu, Jinguo and Su, Weijie and Wang, Junjie and Dong, Xuan and Wang, Wenhai and Li, Bin and Zhou, Jie and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2406.07543},
  year={2024}
}
```

## 📃 License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## 🙏 Acknowledgements

Our code is built with reference to the code of the following projects: [OpenCLIP](https://github.com/mlfoundations/open_clip).
