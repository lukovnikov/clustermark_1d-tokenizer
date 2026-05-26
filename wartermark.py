import hashlib
import math
from typing import List
import torch
from functools import partial
from scipy.stats import norm, binom
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as TF
from huggingface_hub import hf_hub_download
import json


class Watermarker:
    def __init__(self, 
                 num_tokens=None, 
                 prefix=0, 
                 green_fraction=0.5, 
                 red_penalty=5, 
                 vq_clusters=None,
                 token_reconstructor=None,
                 cluster_classifier=None,
                 ):
        
        super().__init__()
        self.num_tokens = num_tokens
        self.prefix = prefix
        self.green_fraction = green_fraction  
        self.red_penalty = red_penalty
        self.vq_clusters = vq_clusters
        self.token_reconstructor = token_reconstructor
        self.cluster_classifier = cluster_classifier

        self.num_tokens = self.num_tokens if self.num_tokens is not None else self.token_reconstructor.num_tokens

        self.mapping = self.vq_clusters
        if self.vq_clusters is None:
            self.mapping = torch.arange(0, self.num_tokens)

        self.precompute_masks()

    def precompute_masks(self):
        """ Precompute masks for every cluster or token"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_entries = self.mapping.max() + 1
        green_masks = torch.zeros(num_entries, self.num_tokens, dtype=torch.bool)      # value of 1 means GREEN token
        green_masks_mapped = torch.zeros(num_entries, num_entries, dtype=torch.bool)  # value of 1 means GREEN token in mapped space
        g = torch.Generator(device=device)

        for i in range(num_entries):
            seq = [self.prefix, i]
            s = ",".join(map(str, seq))
            h = hashlib.sha256(s.encode()).hexdigest()  
            localseed = int(h, 16) % (2**32)
            
            g.manual_seed(localseed)
            
            randvec = torch.rand(num_entries, generator=g, device=device)
            k = int(randvec.shape[0] * self.green_fraction)
            mask = torch.zeros_like(randvec)
            mask[randvec.topk(k).indices] = 1.0
            green_masks_mapped[i] = mask
            mask = mask[self.mapping]
            green_masks[i] = mask

        # green_masks is (num_entries, numtokens), now map to (numtokens, numtokens)
        green_masks = green_masks[self.mapping]     # masks are on token level !!!

        self.green_masks = green_masks.to(device)
        self.green_masks_mapped = green_masks_mapped.to(device)
    
    def get_fraction_greens(self, tokens): # (B, L)
        self.green_masks = self.green_masks.to(tokens.device)
        masks = self.green_masks[tokens]
        isgreen = torch.gather(masks[:, :-1], 2, tokens[:, 1:, None])[:, :, 0]  # (B, L)
        fracgreens = isgreen.float().mean(dim=1)
        return fracgreens
    
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
        if self.red_penalty <= 0:
            return logits
        if ids.shape[1] == 0:
            return logits       # first step, don't watermark
        
        green_mask = self.compute_red_green_masks(output_tokens=ids, logits=logits)
        logits = logits - self.red_penalty * (1 - green_mask.float())

        return logits

    def verify_wm(self, samples):
        B = len(samples)
        with torch.no_grad():
            rec = self.cluster_classifier(samples) if self.cluster_classifier is not None else self.token_reconstructor(samples)

        # COMPUTE METRICS

        numel = rec[0].flatten().shape[0]
        numgreens_thresh = binom.isf(0.01, numel - 1, self.green_fraction)  # threshold for FPR=1%

        fractions_green = []
        pvalues = []
        zstats = []
        tpr1s = []

        rec = rec.view(B, -1)  # flatten the batch dimension
        get_fraction_greens = self.get_fraction_greens if self.cluster_classifier is None else self.get_fraction_greens_mapped
        fractions_green = get_fraction_greens(rec).cpu().numpy().tolist()

        for b in range(B):
            pvalue, zstat = watermark_p_value(fractions_green[b], numel - 1, p0=self.green_fraction)
            tpr1 = fractions_green[b] >= (numgreens_thresh / (numel - 1))  # true positive rate at FPR=1%
            pvalues.append(pvalue)
            zstats.append(zstat)
            tpr1s.append(tpr1)
        return zip(fractions_green, pvalues, zstats, tpr1s), rec


def z_stat(x, n, p0=0.5, continuity=True):
    mu  = n * p0
    sig = math.sqrt(n * p0 * (1 - p0))
    if continuity:
        z = (x - mu + 0.5) / sig   # cc version
    else:
        z = (x - mu) / sig
    p = 1 - norm.cdf(z)            # upper tail
    return z, p


def watermark_p_value(frac_green, total_tokens, p0=0.5):
    num_green = round(frac_green * total_tokens)
    zz, pp = z_stat(num_green, total_tokens, p0=p0, continuity=True)
    binomp = binom.sf(num_green - 1, total_tokens, p0)
    return binomp, zz


def infer_type(value):
    """
    Infers the type of the input string and converts it to float, int, bool, or str.
    
    Parameters:
    value (str): The input string to be converted.
    
    Returns:
    float|int|bool|str: The converted value.
    """
    # Try to convert to int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Try to convert to bool
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    
    # Return as string if no conversion is possible
    return value


class TokenReconstructor:
    def __init__(self, vq_model, mode="llamagen"):
        self.vq_model = vq_model
        self.mode = mode
        if self.mode == "llamagen":
            self.num_tokens = self.vq_model.quantize.n_e
        elif self.mode == "rar":
            self.num_tokens = self.vq_model.quantize.num_embeddings

    def reconstruct_tokens(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.stack([TF.to_tensor(image) for image in images])

        self.vq_model.eval()
        device = next(self.vq_model.parameters()).device
        with torch.no_grad():
            images = images.to(device)
            if self.mode == "llamagen":
                init_h = self.vq_model.encoder(images * 2 - 1)       # h: (B, C, H, W)      # vq_model takes range [-1, 1] ???
                init_h = self.vq_model.quant_conv(init_h)
                init_h, emb_loss, info = self.vq_model.quantize(init_h)
                rec = info[-1]
            elif self.mode == "rar":
                rec = self.vq_model.encode(images)
        return rec

    def __call__(self, images):
        """
        Reconstructs tokens from the given images.
        
        Parameters:
        images (List[Image.Image] or torch.Tensor): List of images or a tensor of images.
        
        Returns:
        torch.Tensor: Reconstructed tokens.
        """
        return self.reconstruct_tokens(images)
    
    @classmethod
    def from_pretrained(cls, vq_model, modelname="rar_xl"):
        if modelname in ["GPT-B", "GPT-L", "GPT-XL"]:
            return LlamaGenTokenReconstructor(vq_model)
        elif modelname in ["rar_b", "rar_l", "rar_xl", "rar_xxl"]:
            return RARTokenReconstructor(vq_model)
        else:
            raise ValueError(f"Unknown mode: {modelname}. Supported modes are 'llamagen' and 'rar'.")


class LlamaGenTokenReconstructor(TokenReconstructor):
    def __init__(self, vq_model):
        super().__init__(vq_model, mode="llamagen")


class RARTokenReconstructor(TokenReconstructor):
    def __init__(self, vq_model):
        super().__init__(vq_model, mode="rar")


def load_model(modelname: str = "rar_l", onlyvae=False, device=0, ckptdir="ckpts"):
    if modelname in ["rar_b", "rar_l", "rar_xl", "rar_xxl"]:
        return load_model_rar(modelsize=modelname, onlyvae=onlyvae, device=device, ckptdir=ckptdir)
    elif modelname in ["GPT-B", "GPT-L", "GPT-XL"]:
        return load_model_llamagen(onlyvae=onlyvae, gpt_model=modelname, device=device, ckptdir=ckptdir)


def load_model_rar(modelsize: str = "rar_l", onlyvae=False, device=0, ckptdir="ckpts"):

    from modeling.rar_wm import RAR
    import demo_util
    from utils.train_utils import create_pretrained_tokenizer

    hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir=ckptdir)

    config = demo_util.get_config("configs/training/generator/rar.yaml")
    config.model.vq_model.pretrained_tokenizer_weight = str(Path(ckptdir) / config.model.vq_model.pretrained_tokenizer_weight)

    config.experiment.generator_checkpoint = str(Path(ckptdir) / f"{modelsize}.bin")
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
    hf_hub_download(repo_id="yucornetto/RAR", filename=f"{modelsize}.bin", local_dir=ckptdir)

    generator = RAR(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    generator.set_random_ratio(0)

    generator.to(device)
    return tokenizer, generator


def load_model_llamagen(
         onlyvae=False,
         downsample_size=16,
         num_classes=1000,
         codebook_embed_dim=8,
         codebook_size=16384,
         gpt_model="GPT-B",
         gpt_type="c2i",
         cls_token_num=1,
         from_fsdp=False,
         precision='bf16',
         compile=False,
         vq_model="VQ-16",
         ckptdir="ckpts",
         device=torch.device("cuda")):

    from tokenizer.tokenizer_image.vq_model import VQ_models
    from autoregressive.models.gpt import GPT_models
        
    model2ckpt = {
        "GPT-XL": ("vq_ds16_c2i.pt", "c2i_XL_384.pt", 384),
        "GPT-B": ("vq_ds16_c2i.pt", "c2i_B_256.pt", 256),
        "GPT-L": ("vq_ds16_c2i.pt", "c2i_L_384.pt", 384),
    }
        
    vq_ckpt, gpt_ckpt, image_size = model2ckpt[gpt_model]
    hf_hub_download(repo_id="FoundationVision/LlamaGen", filename=vq_ckpt, local_dir=ckptdir)
    
    # create and load model
    vq_model = VQ_models[vq_model](
        codebook_size=codebook_size,
        codebook_embed_dim=codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(Path(ckptdir) / vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")
    if onlyvae:
        return vq_model

    hf_hub_download(repo_id="FoundationVision/LlamaGen", filename=gpt_ckpt, local_dir=ckptdir)

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[precision]
    latent_size = image_size // downsample_size
    gpt_model = GPT_models[gpt_model](
        vocab_size=codebook_size,
        block_size=latent_size ** 2,
        num_classes=num_classes,
        cls_token_num=cls_token_num,
        model_type=gpt_type,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(Path(ckptdir) / gpt_ckpt, map_location="cpu")
    if from_fsdp: # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")
    
    if compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    return vq_model, gpt_model


def try_token_reconstruction(mode="rar"):
    """
    Attempts to reconstruct tokens from the given samples using the specified VQ model.
    
    Parameters:
    samples (List[Image.Image]): List of images to reconstruct tokens from.
    vq_model: The VQ model used for token reconstruction.
    mode (str): The mode of reconstruction, either "llamagen" or "rar".
    
    Returns:
    torch.Tensor: Reconstructed tokens.
    """
    # load one image from directory:
    expdir = "experiments_v1/gen_wm_v1_50000samples_rar_xl_64clusters_greenfrac0.5_penalty5"
    image = Image.open(Path(expdir) / "images" / "1.png").convert("RGB")
    tokenseq_original = image.info.get("tokens", None)
    tokenseq_original = torch.tensor(json.loads(tokenseq_original)) if tokenseq_original is not None else None
    vq_model = load_model_rar("rar_xl", onlyvae=True)
    reconstructor = LlamaGenTokenReconstructor(vq_model) if mode == "llamagen" else RARTokenReconstructor(vq_model)
    rec = reconstructor.reconstruct_tokens([image])

    accuracy = (rec.cpu() == tokenseq_original.cpu()).float().mean().item() if tokenseq_original is not None else None
    print(f"Reconstruction accuracy: {accuracy:.4f}" if accuracy is not None else "No original tokens provided for accuracy calculation.")

    # verify watermark
    vq_clusters = torch.load(Path(expdir) / "vq_clusters.pt")
    image_pt = TF.to_tensor(image).unsqueeze(0)  # Convert to tensor and add batch dimension
    wm = Watermarker(num_tokens=reconstructor.num_tokens, prefix=0, green_fraction=0.5, vq_clusters=vq_clusters)
    greenfrac = wm.get_fraction_greens(rec).item()
    print(f"Fraction of green tokens in reconstruction: {greenfrac:.4f}")

    results = wm.verify_wm(image_pt, cluster_classifier=None, token_reconstructor=reconstructor)
    print(f"Watermark verification results: {list(results[0])}")
    return rec



if __name__ == "__main__":
    import fire
    fire.Fire(try_token_reconstruction)