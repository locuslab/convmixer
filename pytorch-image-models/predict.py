import os
import pickle

import torch
from cog import BasePredictor, Input, Path, BaseModel
from PIL import Image
from matplotlib import pyplot as plt

from timm.data import create_transform
from timm.models import create_model
from timm.models.convmixer import _cfg

file_dir = os.path.dirname(os.path.abspath(__file__))
IMAGENET_CLASS_NAME = pickle.load(open(os.path.join(file_dir, "imagenet_class_names.pkl"), "rb"))


class Output(BaseModel):
    plot: Path = None
    Json: str = None

CKPT_PATHS = {
    "convmixer_768_32": "pytorch-image-models/cahced_models/convmixer_768_32_ks7_p7_relu.pth.tar",
    "convmixer_1024_20_ks9_p14": "pytorch-image-models/cahced_models/convmixer_1024_20_ks9_p14.pth.tar",
    "convmixer_1536_20": "pytorch-image-models/cahced_models/convmixer_1536_20_ks9_p7.pth.tar"
}


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        default_cfg = _cfg()
        self.transforms = create_transform(
            input_size=default_cfg['input_size'],
            interpolation=default_cfg['interpolation'],
            mean=default_cfg['mean'],
            std=default_cfg['std'],
            crop_pct=default_cfg['crop_pct'],
        )

    def predict(
        self,
        model_name: str = Input(
            description="Model name to use",
            default="convmixer_768_32",
            choices=["convmixer_768_32", "convmixer_1024_20_ks9_p14", "convmixer_1536_20"],
        ),
        input_image: Path = Input(
            description="Image to be classified"
        ),
        output_format: str = Input(
            description="Recieve outputs as Json or a plot", default="Plot", choices=['Json', 'Plot']
        ),
    ) -> Output:
        assert(os.path.exists(CKPT_PATHS[model_name]))
        model = create_model(
            model_name,
            num_classes=1000,
            in_chans=3,
            pretrained=False,
            checkpoint_path=CKPT_PATHS[model_name]
        ).cuda()

        model.eval()

        with torch.no_grad():
            image = Image.open(str(input_image)).convert("RGB")
            image = self.transforms(image)[None,].cuda()

            labels = model(image)

            top_k = labels.topk(10)
            scores = torch.softmax(top_k[0][0], dim=0).cpu().numpy()
            classes_ids = top_k[1].cpu().numpy()[0]
            classes = [IMAGENET_CLASS_NAME[id].split(",")[0] for id in classes_ids]

            if output_format == "Json":
                return Output(Json=str(dict(zip(classes, scores))))
            else:
                output_path = "topk_plot.png"
                plot_topk_classes(classes[::-1], scores[::-1], output_path)
                return Output(plot=Path(output_path))


def plot_topk_classes(names, scores, path):
    n = len(names)
    plt.figure(figsize=[4, 4])
    plt.barh(range(n), scores, align="center")
    plt.yticks(range(n), names)
    plt.tight_layout()

    plt.savefig(path)
