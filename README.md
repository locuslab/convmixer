# Patches Are All You Need? 🤷
This repository contains an implementation of ConvMixer for the ICLR 2022 submission ["Patches Are All You Need?"](https://openreview.net/forum?id=TVHS5Y4dNvM).

### Code overview
The most important code is in `convmixer.py`. We trained ConvMixers using the `timm` framework, which we copied from [here](http://github.com/rwightman/pytorch-image-models).

__**Update:**__ ConvMixer is now integrated into the [`timm` framework itself](https://github.com/rwightman/pytorch-image-models). You can see the PR [here](https://github.com/rwightman/pytorch-image-models/pull/910).

Inside `pytorch-image-models`, we have made the following modifications. (Though one could look at the diff, we think it is convenient to summarize them here.)

- Added ConvMixers
  - added `timm/models/convmixer.py`
  - modified `timm/models/__init__.py`
- Added "OneCycle" LR Schedule
  - added `timm/scheduler/onecycle_lr.py`
  - modified `timm/scheduler/scheduler.py`
  - modified `timm/scheduler/scheduler_factory.py`
  - modified `timm/scheduler/__init__.py`
  - modified `train.py` (added two lines to support this LR schedule)

We are confident that the use of the OneCycle schedule here is not critical, and one could likely just as well
train ConvMixers with the built-in cosine schedule.

### Evaluation
We provide some model weights below:

| Model Name | Kernel Size | Patch Size | File Size |
|------------|:-----------:|:----------:|----------:|
|[ConvMixer-1536/20](https://github.com/tmp-iclr/convmixer/releases/download/v1.0/convmixer_1536_20_ks9_p7.pth.tar)| 9 | 7 | 207MB |
|[ConvMixer-768/32](https://github.com/tmp-iclr/convmixer/releases/download/v1.0/convmixer_768_32_ks7_p7_relu.pth.tar)\*| 7 | 7 | 85MB |
|[ConvMixer-1024/20](https://github.com/tmp-iclr/convmixer/releases/download/v1.0/convmixer_1024_20_ks9_p14.pth.tar)| 9 | 14 | 98MB |

\* **Important:** ConvMixer-768/32 here uses ReLU instead of GELU, so you would have to change `convmixer.py` accordingly (we will fix this later).

You can evaluate ConvMixer-1536/20 as follows:

```
python validate.py --model convmixer_1536_20 --b 64 --num-classes 1000 --checkpoint [/path/to/convmixer_1536_20_ks9_p7.pth.tar] [/path/to/ImageNet1k-val]
```

You should get a `81.37%` accuracy.

### Training
If you had a node with 10 GPUs, you could train a ConvMixer-1536/20 as follows (these are exactly the settings we used):

```
sh distributed_train.sh 10 [/path/to/ImageNet1k] 
    --train-split [your_train_dir] 
    --val-split [your_val_dir] 
    --model convmixer_1536_20 
    -b 64 
    -j 10 
    --opt adamw 
    --epochs 150 
    --sched onecycle 
    --amp 
    --input-size 3 224 224
    --lr 0.01 
    --aa rand-m9-mstd0.5-inc1 
    --cutmix 0.5 
    --mixup 0.5 
    --reprob 0.25 
    --remode pixel 
    --num-classes 1000 
    --warmup-epochs 0 
    --opt-eps=1e-3 
    --clip-grad 1.0
```

We also included a ConvMixer-768/32 in timm/models/convmixer.py (though it is simple to add more ConvMixers). We trained that one with the above settings but with 300 epochs instead of 150 epochs.

In the near future, we will upload weights.

The tweetable version of ConvMixer, which requires `from torch.nn import *`:

```
def ConvMixr(h,d,k,p,n):
 S,C,A=Sequential,Conv2d,lambda x:S(x,GELU(),BatchNorm2d(h))
 R=type('',(S,),{'forward':lambda s,x:s[0](x)+x})
 return S(A(C(3,h,p,p)),*[S(R(A(C(h,h,k,groups=h,padding=k//2))),A(C(h,h,1))) for i in range(d)],AdaptiveAvgPool2d((1,1)),Flatten(),Linear(h,n))
```
