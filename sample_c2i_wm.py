import torch

from PIL import Image, PngImagePlugin
import numpy as np
import demo_util
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from torchvision.transforms.functional import to_pil_image, to_tensor
from modeling.rar_wm import RAR
import hashlib

import fire, time, tqdm, json, math, itertools
from pathlib import Path


CKPTDIR = "ckpts"
# DEBUG = True
DEBUG = False


def reconstruct_tokens(images, tokenizer):
    with torch.no_grad():
        tokenizer.eval()
        device = next(tokenizer.parameters()).device
        images = images.to(device)
        input_tokens = tokenizer.encode(images)
    return input_tokens


def load_model(modelsize: str = "rar_l", onlyvae=False, device=0):
    hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir=CKPTDIR)

    config = demo_util.get_config("configs/training/generator/rar.yaml")
    config.model.vq_model.pretrained_tokenizer_weight = str(Path(CKPTDIR) / config.model.vq_model.pretrained_tokenizer_weight)

    config.experiment.generator_checkpoint = str(Path(CKPTDIR) / f"{modelsize}.bin")
    config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[modelsize]
    config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[modelsize]
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[modelsize]

    # download the maskgit-vq tokenizer
    # maskgit-vq as tokenizer
    tokenizer = create_pretrained_tokenizer(config)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer.to(device)

    if onlyvae:
        return tokenizer
    
    # download the rar generator weight
    hf_hub_download(repo_id="yucornetto/RAR", filename=f"{modelsize}.bin", local_dir=CKPTDIR)

    generator = RAR(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    generator.set_random_ratio(0)

    generator.to(device)
    return tokenizer, generator


class Watermarker:
    def __init__(self, wm_seed_prefix=0, 
                 wm_red_penalty=10, 
                 wm_green_fraction=0.5, 
                 numtokens=None, 
                 vq_clusters=None,
                 ):
        
        super().__init__()
        self.wm_seed_prefix = wm_seed_prefix
        self.wm_red_penalty = wm_red_penalty
        self.wm_green_fraction = wm_green_fraction  
        self.vq_clusters = vq_clusters
        self.numtokens = numtokens
        self.mapping = self.vq_clusters
        if self.vq_clusters is None:
            self.mapping = torch.arange(0, self.numtokens)

        self.precompute_masks()

    def precompute_masks(self):
        """ Precompute masks for every cluster or token"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_entries = self.mapping.max() + 1
        green_masks = torch.zeros(num_entries, self.numtokens, dtype=torch.bool)      # value of 1 means GREEN token
        green_masks_mapped = torch.zeros(num_entries, num_entries, dtype=torch.bool)  # value of 1 means GREEN token in mapped space
        g = torch.Generator(device=device)

        for i in range(num_entries):
            seq = [self.wm_seed_prefix, i]
            s = ",".join(map(str, seq))
            h = hashlib.sha256(s.encode()).hexdigest()  
            localseed = int(h, 16) % (2**32)
            
            g.manual_seed(localseed)
            
            randvec = torch.rand(num_entries, generator=g, device=device)
            k = int(randvec.shape[0] * self.wm_green_fraction)
            mask = torch.zeros_like(randvec)
            mask[randvec.topk(k).indices] = 1.0
            green_masks_mapped[i] = mask
            mask = mask[self.mapping]
            green_masks[i] = mask

        # green_masks is (num_entries, numtokens), now map to (numtokens, numtokens)
        green_masks = green_masks[self.mapping]     # masks are on token level !!!

        self.green_masks = green_masks.to(device)
        self.green_masks_mapped = green_masks_mapped.to(device)

    def get_num_greens(self, tokens):  # (B, L)
        self.green_masks = self.green_masks.to(tokens.device)
        masks = self.green_masks[tokens]
        isgreen = torch.gather(masks[:, :-1], 2, tokens[:, 1:, None])[:, :, 0]  # (B, L)
        numgreens = isgreen.float().sum(dim=1)
        return numgreens
    
    def get_fraction_greens(self, tokens): # (B, L)
        self.green_masks = self.green_masks.to(tokens.device)
        masks = self.green_masks[tokens]
        isgreen = torch.gather(masks[:, :-1], 2, tokens[:, 1:, None])[:, :, 0]  # (B, L)
        fracgreens = isgreen.float().mean(dim=1)
        return fracgreens

    def get_num_greens_mapped(self, tokens):  # (B, L)
        self.green_masks_mapped = self.green_masks_mapped.to(tokens.device)
        masks = self.green_masks_mapped[tokens]
        isgreen = torch.gather(masks[:, :-1], 2, tokens[:, 1:, None])[:, :, 0]  # (B, L)
        numgreens = isgreen.float().sum(dim=1)
        return numgreens
    
    def get_fraction_greens_mapped(self, tokens): # (B, L)
        self.green_masks_mapped = self.green_masks_mapped.to(tokens.device)
        masks = self.green_masks_mapped[tokens]
        isgreen = torch.gather(masks[:, :-1], 2, tokens[:, 1:, None])[:, :, 0]  # (B, L)
        fracgreens = isgreen.float().mean(dim=1)
        return fracgreens

    def compute_red_green_masks(self, output_tokens=None, logits=None):
        device = logits.device
        self.green_masks = self.green_masks.to(device)

        B, vocab_size = logits.shape
        mask = torch.zeros(B, vocab_size, dtype=torch.bool, device=device)
        
        last_tokens = output_tokens[:, -1]     # (B,)
        
        mask = self.green_masks[last_tokens]
        mask = mask.to(logits.device)

        # return value of one means GREEN token
        return mask
        
    def apply_watermark(self, logits, ids, step, condition):
        if self.wm_red_penalty <= 0:
            return logits
        if ids.shape[1] == 0:
            return logits       # first step, don't watermark
        
        green_mask = self.compute_red_green_masks(output_tokens=ids, logits=logits)
        logits = logits - self.wm_red_penalty * (1 - green_mask.float())

        return logits

    # # return value of one means GREEN token
    # def _original_compute_red_green_masks(self, output_tokens=None, logits=None):
    #     device = logits.device
    #     B, vocab_size = logits.shape
    #     mask = torch.zeros(B, vocab_size, dtype=torch.bool, device=device)
        
    #     output_tokens = output_tokens[-1]
    #     num_clusters = vq_clusters.max() + 1 if vq_clusters is not None else vocab_size
        
    #     g = torch.Generator(device=device)
        
    #     for b in range(B):
    #         if output_tokens.shape[-1] > 0:
    #             seed_cond = output_tokens[b, -1].item()
    #             seed_cond_cluster = vq_clusters[seed_cond].item() if vq_clusters is not None else seed_cond
    #             seq = [wm_seed_prefix, seed_cond_cluster]
    #             s = ",".join(map(str, seq))
    #             h = hashlib.sha256(s.encode()).hexdigest()  
    #             localseed = int(h, 16) % (2**32)
                
    #             g.manual_seed(localseed)
                
    #             vec_clusters = torch.rand(num_clusters, generator=g, device=device)
    #             k = int(vec_clusters.shape[0] * wm_green_fraction)
    #             clustermask = torch.zeros_like(vec_clusters)
    #             clustermask[vec_clusters.topk(k).indices] = 1.0
    #             mask[b] = clustermask[vq_clusters] if vq_clusters is not None else clustermask
    #             # vec = vec_clusters[vq_clusters] if vq_clusters is not None else vec_clusters
    #             # mask[b] = vec < vec.quantile(wm_green_fraction)
                    
    #     mask = mask.to(logits.device)
    #     return mask


def infer(tokenizer, generator, 
          n: int = 16,
          class_labels: list = None,
          cfg_scale: float = 16.0,
          cfg_pow: int = 1,
          randomize_temperature: float = 1.0,
            watermarker=None,
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
        watermarker=watermarker,
        num_sample_steps=num_sample_steps)
    
    generated_images = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    generated_images = torch.clamp(generated_images, 0.0, 1.0)
    pilimages = [to_pil_image(generated_image.cpu()) for generated_image in generated_images]

    if DEBUG:
        recons_tokens = reconstruct_tokens(generated_images, tokenizer)
        recons_accuracy = (recons_tokens == generated_tokens).float().mean().item()

        reloaded_images = torch.stack([to_tensor(pilimage) for pilimage in pilimages], 0)
        recons_pilimages = reconstruct_tokens(reloaded_images, tokenizer)
        recons_pil_accuracy = (recons_tokens == generated_tokens).float().mean().item()

        num_greens_gen = watermarker.get_num_greens(generated_tokens)
        num_greens_recons = watermarker.get_num_greens(recons_tokens)
        num_greens_recons_pil = watermarker.get_num_greens(recons_pilimages)

        reconstructed_images = tokenizer.decode_tokens(
            recons_tokens.view(recons_tokens.shape[0], -1)
        )
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        reconstructed_images = [to_pil_image(reconstructed_image.cpu()) for reconstructed_image in reconstructed_images]

        for i, (reconstructed_image, generated_image) in enumerate(zip(reconstructed_images, pilimages)):
            reconstructed_image.save(f"testimgs/reconstructed_{i}.png")
            generated_image.save(f"testimgs/generated_{i}.png")

    return pilimages, generated_tokens


def main(modelsize: str = "rar_xl",    # "rar_b", "rar_l", "rar_xl", "rar_xxl"
         num_samples=2000, 
         batsize=-1,
         num_clusters=64,            # 8, 32, 64, 128
         load_clusters="clusters_balanced",
         seed=420, 
         wm_seed_prefix=0,
         wm_red_penalty=5,         # 5, 2, 1
         wm_green_fraction=0.25,     # 0.5, 0.25
         savedir="experiments_rebuttal",
         expprefix="gen_wm_v1",
         overwrite=DEBUG,
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


    savedir = Path(savedir) / f"{expprefix}_{num_samples}samples_{modelsize}_{num_clusters}clusters_greenfrac{wm_green_fraction}_penalty{wm_red_penalty}_prefix{wm_seed_prefix}"
    if not savedir.exists():
        savedir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Warning: {savedir} already exists, will overwrite files in this directory.")
        if not overwrite:
            print("Set overwrite=True to overwrite files in this directory.")
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

    # create clusters of VQ vectors
    if num_clusters > 0:
        # load precomputed clusters
        clusterfile = Path(load_clusters) / f"balanced_kmeans_{num_clusters}.pt"
        print("loading clusters from", clusterfile)
        vq_clusters = torch.load(clusterfile, map_location="cpu")
        # compute cluster sizes:
        clustersizes = (vq_clusters[:, None] == torch.arange(num_clusters, device=vq_clusters.device)[None, :]).sum(0)
        print("Max and min cluster sizes: ", clustersizes.max(), clustersizes.min())
        print("Num tokens:", vq_clusters.shape)
        torch.save(vq_clusters, savedir / "vq_clusters.pt")
    else:
        print("No clusters specified.")
        vq_clusters = None

    watermarker = Watermarker(
        wm_seed_prefix=wm_seed_prefix,
        wm_red_penalty=wm_red_penalty,
        wm_green_fraction=wm_green_fraction,
        numtokens=tokenizer.quantize.num_embeddings,
        vq_clusters=vq_clusters,
    )

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
                watermarker=watermarker,
                device=device,
                seed=seed)
            infer_time = time.time() - t1

        fracgreen = watermarker.get_fraction_greens(tokens)
        
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

            if DEBUG:
                reloaded = Image.open(filename)
                reloaded_tokens = reconstruct_tokens(to_tensor(reloaded).unsqueeze(0), tokenizer)
                fracgreens = watermarker.get_fraction_greens(reloaded_tokens)

    print(f"generated {num_generated} images in total.")


if __name__ == "__main__":
    fire.Fire(main)