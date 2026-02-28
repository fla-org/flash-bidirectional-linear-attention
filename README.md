<div align="center">

# Flash Bidirectional Linear Attention

</div>

The aim of this repository is to implement **bidirectional linear attention** for **non-causal** modeling using Triton. Contributions and suggestions are welcome!

<div align="center">
  <img width="600" alt="image" src="https://res.cloudinary.com/dunty6aot/image/upload/v1735544947/387246938-cd89a618-5d54-41b7-9055-36ba28b29fbd-2_tailvo.png">
</div>


# Update
* [2026/02] Update PISA
* [2025/02] Update PolaFormer
* [2024/12] Update `simple_la`, a simple form of `linear_attn` without the norm term.

# Models
Roughly sorted according to the timeline supported in `flash_bla`

| Year    | Model     | Title                                                                  | Paper                                     | Code                                                          | `fla` impl                                                                                                           |
| :------ | :-------- | :--------------------------------------------------------------------- | :---------------------------------------: | :-----------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| 2024 | Linfusion | LinFusion: 1 GPU, 1 Minute, 16K Image                                  | [arxiv](https://arxiv.org/abs/2409.02097) | [official](https://github.com/Huage001/LinFusion)             | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/flash_bla/layers/linfusion.py)           |
| 2024 | MLLA      | Demystify Mamba in Vision: A Linear Attention Perspective              | [arxiv](https://arxiv.org/abs/2405.16605) | [official](https://github.com/LeapLabTHU/MLLA)                | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/flash_bla/layers/mlla.py)                |
| 2023 | Focused-LA| FLatten Transformer: Vision Transformer using Focused Linear Attention | [arxiv](https://arxiv.org/abs/2308.00442) | [official](https://github.com/LeapLabTHU/FLatten-Transformer) | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/flash_bla/layers/focused_la.py)          |
| 2025 | PolaFormer| PolaFormer: Polarity-aware Linear Attention for Vision Transformers    | [arxiv](https://arxiv.org/abs/2501.15061) | [official](https://github.com/ZacharyMeng/PolaFormer) | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/flash_bla/layers/polaformer.py)                 |
| 2025 | RALA| Breaking the Low-Rank Dilemma of Linear Attention   | [arxiv](https://arxiv.org/abs/2411.07635) | [official](https://github.com/qhfan/RALA) | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/fbi_la/layers/rala.py)   
| 2026 | PISA      | PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers               | [arxiv](https://arxiv.org/abs/2602.01077) | [official](https://github.com/xie-lab-ml/piecewise-sparse-attention)               | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/flash_bla/ops/piecewise_attn/kernel.py)                

# Usage

## Installation
``` shell
git clone https://github.com/fla-org/flash-bidirectional-linear-attention.git
pip install -e flash-bidirectional-linear-attention/.
```

## Integrated Models
This library has integrated some models, which can be called directly. Taking [LinFusion](https://github.com/Huage001/LinFusion) as an example:
``` python
import torch
from diffusers import AutoPipelineForText2Image
from flash_bla.models import LinFusion

sd_repo = "Lykon/dreamshaper-8"

pipeline = AutoPipelineForText2Image.from_pretrained(
    sd_repo, torch_dtype=torch.float16, variant="fp16"
).to(torch.device("cuda"))

linfusion = LinFusion.construct_for(pipeline)

image = pipeline(
    "An astronaut floating in space. Beautiful view of the stars and the universe in the background.",
    generator=torch.manual_seed(123)
).images[0]
```

# Benchmarks
Profiled on the A800-80G GPU.
```shell
B8-H16-D64:
    T  torch_la_fwd  flash_bla_fwd  torch_sdpa_fwd  torch_la_bwd  flash_bla_bwd  torch_sdpa_bwd
 1024      0.083968       0.068608        0.073728      0.476160       0.378880        0.405504
 4096      0.178176       0.083968        0.784384      1.018880       0.444416        3.175424
16384      0.549888       0.283648       11.750400      3.556352       1.566720       44.189184
32768      1.034240       0.550912       47.788033      6.864896       3.040256      175.127548
```


# Acknowledgments
Thanks to the following repositories for their inspiration:
- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)

