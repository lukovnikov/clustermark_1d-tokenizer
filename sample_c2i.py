import torch

from PIL import Image, PngImagePlugin
import numpy as np
import demo_util
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from torchvision.transforms.functional import to_pil_image, to_tensor

import fire, time, tqdm, json, math, itertools
from pathlib import Path


CKPTDIR = "ckpts"


def load_model(modelsize: str = "rar_l", device=0):

    # download the maskgit-vq tokenizer
    hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir=CKPTDIR)
    # download the rar generator weight
    hf_hub_download(repo_id="yucornetto/RAR", filename=f"{modelsize}.bin", local_dir=CKPTDIR)

    # load config
    config = demo_util.get_config("configs/training/generator/rar.yaml")
    config.experiment.generator_checkpoint = Path(CKPTDIR) / f"{modelsize}.bin"
    config.model.vq_model.pretrained_tokenizer_weight = str(Path(CKPTDIR) / config.model.vq_model.pretrained_tokenizer_weight)
    config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[modelsize]
    config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[modelsize]
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[modelsize]


    device = "cuda"
    # maskgit-vq as tokenizer
    tokenizer = create_pretrained_tokenizer(config)
    generator = demo_util.get_rar_generator(config)
    tokenizer.to(device)
    generator.to(device)
    return tokenizer, generator


def infer(tokenizer, generator, 
          n: int = 16,
          class_labels: list = None,
          cfg_scale: float = 16.0,
          cfg_pow: int = 1,
          randomize_temperature: float = 1.0,
          device: torch.device = torch.device("cuda"),

            guidance_decay="constant",
            softmax_temperature_annealing=False,
            num_sample_steps=8,
          seed: int = 0):
    labels = torch.tensor(class_labels, dtype=torch.int64, device=device) if class_labels is not None else None

    generator.eval()
    tokenizer.eval()

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=cfg_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=cfg_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        num_sample_steps=num_sample_steps)
    
    generated_images = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )

    generated_images = torch.clamp(generated_images, 0.0, 1.0)
    pilimages = [to_pil_image(generated_image.cpu()) for generated_image in generated_images]

    return pilimages, generated_tokens


def main(modelsize: str = "rar_xl",    # "rar_b", "rar_l", "rar_xl", "rar_xxl"
         num_samples=50000, 
         batsize=-1,
        #  num_clusters=8,            # 8, 32, 64, 128
        #  load_clusters="clusters_balanced",
         seed=420, 
        #  wm_seed_prefix=0,
        #  wm_red_penalty=10,         # 5, 2, 1
        #  wm_green_fraction=0.5,     # 0.5, 0.25
         savedir="experiments_v1",
         expprefix="gen_clean_v1",
         overwrite=False,
         device=0):
    batsize = {"rar_b": 64, "rar_l": 64, "rar_xl": 50, "rar_xxl": 32}[modelsize] if batsize < 0 else batsize

    cfg_scale = {"rar_b": 16.0, "rar_l": 15.5, "rar_xl": 6.9, "rar_xxl": 8.0}[modelsize]
    cfg_pow = {"rar_b": 2.75, "rar_l": 2.5, "rar_xl": 1.5, "rar_xxl": 1.2}[modelsize]
    randomize_temperature = {"rar_b": 1.0, "rar_l": 1.02, "rar_xl": 1.02, "rar_xxl": 1.02}[modelsize]

    args = locals().copy()
    print(json.dumps(args, indent=4))

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_grad_enabled(False)


    savedir = Path(savedir) / f"{expprefix}_{num_samples}samples_{modelsize}"
    if not savedir.exists():
        savedir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Warning: {savedir} already exists, will overwrite files in this directory.")
        if not overwrite:
            return
    if not (savedir / "images").exists():
        (savedir / "images").mkdir(parents=True, exist_ok=True)
        
    json.dump(args, open(savedir / "args.json", "w"), indent=4)


    device = torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")

    t = time.time()
    print("loading model...")
    tokenizer, generator = load_model(modelsize=modelsize, device=device)
    t = time.time() - t
    print(f"model loaded in {t:.2f} seconds.")

    num_classes = 1000
    classes = itertools.cycle(list(range(num_classes)))

    num_generated = 0

    for _ in tqdm.tqdm(range(math.ceil(num_samples / batsize))):
    # while num_generated < num_samples:
        with torch.no_grad():
            class_labels = [next(classes) for _ in range(batsize)]
            t1 = time.time()
            samples, tokens = infer(
                tokenizer, generator,
                n=batsize,
                class_labels=class_labels,
                cfg_scale=cfg_scale,
                cfg_pow=cfg_pow,
                randomize_temperature=randomize_temperature,
                device=device,
                seed=seed)
            infer_time = time.time() - t1
        
        filenames = []
        for i, (sample, classlabel) in enumerate(zip(samples, class_labels)):
            meta = PngImagePlugin.PngInfo()
            meta.add_text("class_label", str(classlabel))
            meta.add_text("infer_time", str(infer_time))
            meta.add_text("tokens", str(tokens[i].detach().cpu().numpy().tolist()))
            for k, v in args.items():
                meta.add_text(k, str(v))
            filename = Path(f"{savedir}/images/{num_generated}.png")
            sample.save(filename, "PNG", pnginfo=meta)
            filenames.append(filename.name)
            num_generated += 1
            if num_generated >= num_samples:
                break

    print(f"generated {num_generated} images in total.")


if __name__ == "__main__":
    fire.Fire(main)