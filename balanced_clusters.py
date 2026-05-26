import sys, fire
from wartermarks.balanced_clusters import main_codebook, main_tokenpred


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main_codebook()
    else:
        fire.Fire({
            "codebook": main_codebook,
            "tokenpred": main_tokenpred,
        })
