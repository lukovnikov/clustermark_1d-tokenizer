import sys
import fire
from wartermarks.evaluate import eval_one, eval_many, eval_roc_one, eval_roc_many, eval_cleanfid_one, eval_cleanfid_many


DEBUG = True
DEBUG = False
DIR = "experiments_v1/gen_wm_v1_50000samples_rar_xl_64clusters_greenfrac0.5_penalty5"
CLUSTERPRED = "experiments_v1/gen_clean_v1_100000samples_rar_xl/checkpoints-encoder/encoder_epoch_30.pt"
TOKENPRED = "experiments_v1/gen_clean_v1_100000samples_rar_xl/checkpoints-tokenpred_cp_24epochs/encoder_epoch_24.pt"


def _eval_roc_one(
        # dir="experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2/",
        dir="experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3",
        # posfile="results_v3_tokpred_evalfirst2000_perturb=fullchallenge.json",
        posfile="results_v3_tokpred_evalfirst2000_perturb=allregen.json",
        negfile="results_v3_tokpred_evalfirst2000_perturb=negative.json",
         overwrite=False,
         do_histograms=True,
         ):
    # print("Running eval_roc_one with parameters:")
    # print(f"dir: {dir}, posfile: {posfile}, negfile: {negfile}, overwrite: {overwrite}, do_histograms: {do_histograms}")
    return eval_roc_one(
        dir=dir,
        posfile=posfile,
        negfile=negfile,
        overwrite=overwrite,
        do_histograms=do_histograms
    )


def _eval_roc_many(expdir="experiments_v1_prefixes/",
                    dryrun=False,
                    posfile="results_v3_clpred_evalfirst2000_perturb=fullchallenge.json",
                    negfile="results_v3_clpred_evalfirst2000_perturb=negative.json",
                    overwrite=False,
                    do_histograms=True,
                    **kwargs,
         ):
    # print("Running eval_roc_many with parameters:")
    # print(f"expdir: {expdir}, dryrun: {dryrun}, posfile: {posfile}, negfile: {negfile}, overwrite: {overwrite}, do_histograms: {do_histograms}")
    return eval_roc_many(
        expdir=expdir,
        dryrun=dryrun,
        posfile=posfile,
        negfile=negfile,
        overwrite=overwrite,
        do_histograms=do_histograms,
        **kwargs
    )


def _eval_many(expdir="experiments_v1_prefixes/",
                    batsize=64,
                    numworkers=8,
                    dryrun=False,
                    device=0,
                    evalfirstk=2000,
                    evaloffset=0,
                    perturbationset="full",
                    resultsfile="",
                    summaryfile="",
                    useclusterpredictor=False,
                    clusterpredictor=CLUSTERPRED,
                    usetokenpredictor=False,
                    tokenpredictor=TOKENPRED,
                    eval_seed_prefix=-1,
                    eval_green_fraction=-1,
                    seed=89,
                    imgdir=None,
                    **kwargs,
         ):
    # print("Running eval_many with parameters:")
    # print(f"expdir: {expdir}, batsize: {batsize}, numworkers: {numworkers}, dryrun: {dryrun}, device: {device}, evalfirstk: {evalfirstk}, perturbationset: {perturbationset}, resultsfile: {resultsfile}, summaryfile: {summaryfile}, useclusterpredictor: {useclusterpredictor}, clusterpredictor: {clusterpredictor}, eval_seed_prefix: {eval_seed_prefix}, eval_green_fraction: {eval_green_fraction}, seed: {seed}, imgdir: {imgdir}")
    return eval_many(
        expdir=expdir,
        batsize=batsize,
        numworkers=numworkers,
        dryrun=dryrun,
        device=device,
        evalfirstk=evalfirstk,
        evaloffset=evaloffset,
        perturbationset=perturbationset,
        resultsfile=resultsfile,
        summaryfile=summaryfile,
        useclusterpredictor=useclusterpredictor,
        clusterpredictor=clusterpredictor,
        usetokenpredictor=usetokenpredictor,
        tokenpredictor=tokenpredictor,
        eval_seed_prefix=eval_seed_prefix,
        eval_green_fraction=eval_green_fraction,
        seed=seed,
        imgdir=imgdir,
        **kwargs
    )


def _eval_one(
        #  dir="experiments_v1/gen_wm_v1_50000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3",
        #  dir="experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2",
        dir="experiments_rebuttal7/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix0 copy",
         batsize=32,
         numworkers=0 if DEBUG else 8,
         evalfirstk=2000,
         evaloffset=0,
         perturbationset="fullchallenge",
         resultsfile="",
         summaryfile="",
         useclusterpredictor=False,
         clusterpredictor="clusterpredictor.pt",
        usetokenpredictor=False,
        # tokenpredictor="experiments_v1/gen_clean_v1_100000samples_rar_xl/checkpoints-tokenpred_cp_30epochs/encoder_epoch_28.pt",
        tokenpredictor="tokenpredictor.pt",
         wm_prefix=-1 if not DEBUG else 0,
         wm_green_fraction=-1 if not DEBUG else 0.5,
         seed=89,
         device=0,
         imgdir=None,
        #  imgdir="experiments_v1/gen_clean_v1_50000samples_rar_xl/",
         tag="",
         overwrite=False,
         ):
# def _eval_one(
#          dir="experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3",
#         #  dir="experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2",
#          batsize=64,
#          numworkers=0 if DEBUG else 8,
#          evalfirstk=2000,
#          perturbationset="regen",
#          resultsfile="",
#          summaryfile="",
#          useclusterpredictor=True,
#          clusterpredictor=CLUSTERPRED,
#         usetokenpredictor=False,
#         tokenpredictor="experiments_v1/gen_clean_v1_100000samples_rar_xl/checkpoints-tokenpred_cp_30epochs/encoder_epoch_28.pt",
#          wm_prefix=-1 if not DEBUG else 0,
#          wm_green_fraction=-1 if not DEBUG else 0.5,
#          seed=89,
#          device=0,
#          imgdir=None,
#          tag="",
#          overwrite=False,
#          ):

# def _eval_one(
#          dir="experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3",
#         #  dir="experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2",
#          batsize=64,
#          numworkers=0 if DEBUG else 8,
#          evalfirstk=2000,
#          perturbationset="regen",
#          resultsfile="",
#          summaryfile="",
#          useclusterpredictor=False,
#          clusterpredictor=CLUSTERPRED,
#         usetokenpredictor=True,
#         tokenpredictor="experiments_v1/gen_clean_v1_100000samples_rar_xl/checkpoints-tokenpred_cp_30epochs/encoder_epoch_28.pt",
#          wm_prefix=-1 if not DEBUG else 0,
#          wm_green_fraction=-1 if not DEBUG else 0.5,
#          seed=89,
#          device=0,
#          imgdir=None,
#          tag="",
#          overwrite=False,
#          ):
    
    # print("Running eval_one with parameters:")
    # print(f"dir: {dir}, batsize: {batsize}, numworkers: {numworkers}, evalfirstk: {evalfirstk}, perturbationset: {perturbationset}, resultsfile: {resultsfile}, summaryfile: {summaryfile}, useclusterpredictor: {useclusterpredictor}, clusterpredictor: {clusterpredictor}, wm_prefix: {wm_prefix}, wm_green_fraction: {wm_green_fraction}, seed: {seed},
    # device: {device}, imgdir: {imgdir}, tag: {tag}, overwrite: {overwrite}")
    return eval_one(
        dir=dir,
        batsize=batsize,
        numworkers=numworkers,
        evalfirstk=evalfirstk,
        evaloffset=evaloffset,
        perturbationset=perturbationset,
        resultsfile=resultsfile,
        summaryfile=summaryfile,
        useclusterpredictor=useclusterpredictor,
        clusterpredictor=clusterpredictor,
        usetokenpredictor=usetokenpredictor,
        tokenpredictor=tokenpredictor,
        wm_prefix=wm_prefix,
        wm_green_fraction=wm_green_fraction,
        seed=seed,
        device=device,
        imgdir=imgdir,
        tag=tag,
        overwrite=overwrite,
    )


def _eval_cleanfid_one(generated_path,
                real_path='/USERSPACE/DATASETS/imagenet/val',
                batch_size=512,
                num_workers=16,
                device=0,
                mode='clean',
                model_name='inception_v3',
                verbose=False,
                resize_size=256,
                nosave=False,
                ):
    # print("Running eval_cleanfid_one with parameters:")
    # print(f"generated_path: {generated_path}, real_path: {real_path}, batch_size: {batch_size}, num_workers: {num_workers}, device: {device}, mode: {mode}, model_name: {model_name}, verbose: {verbose}, resize_size: {resize_size}, nosave: {nosave}")
    return eval_cleanfid_one(
        generated_path=generated_path,
        real_path=real_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,              
        mode=mode,
        model_name=model_name,
        verbose=verbose,
        resize_size=resize_size,
        nosave=nosave,
    )


def _eval_cleanfid_many(expdir="experiments_v1/",
                    real_path='/USERSPACE/DATASETS/imagenet/val',
                    batch_size=128,
                    num_workers=4,
                    mode='clean',
                    model_name='inception_v3',
                    verbose=False,
                    resize_size=256,
                    nosave=False,
                    dryrun=False,
                    overwrite=False,
                    device=0,
                    **kwargs,
         ):
    # print("Running eval_cleanfid_many with parameters:")
    # print(f"expdir: {expdir}, real_path: {real_path}, batch_size: {batch_size}, num_workers: {num_workers}, mode: {mode}, model_name: {model_name}, verbose: {verbose}, resize_size: {resize_size}, nosave: {nosave}, dryrun: {dryrun}, overwrite: {overwrite}, device: {device}")
    return eval_cleanfid_many(
        expdir=expdir,
        real_path=real_path,
        batch_size=batch_size,
        num_workers=num_workers,
        mode=mode,
        model_name=model_name,
        verbose=verbose,
        resize_size=resize_size,
        nosave=nosave,
        dryrun=dryrun,
        overwrite=overwrite,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No arguments provided. Running eval_many with default parameters.")
        _eval_one()
        # eval_roc_many()
    else:
        fire.Fire(
            {
                "one": _eval_one, 
                "many": _eval_many, 
                "roc_one": _eval_roc_one,
                "roc_many": _eval_roc_many,
                "cleanfid_one": _eval_cleanfid_one,
                "cleanfid_many": _eval_cleanfid_many,
            }
        )