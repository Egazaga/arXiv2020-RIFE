import os
import shutil
from glob import glob

from tqdm import tqdm

from inference_imgs import infer_rife, get_rife_model
from scene_detection import find_scenes


def rife_stage():
    in_path = "input/"
    out_path = "output/"
    first_inter = "output/first_inter/"
    shutil.rmtree(first_inter, ignore_errors=True)
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(first_inter, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    slices = find_scenes(in_path, return_slices=True)
    print(slices)
    imgs_paths = sorted(glob(in_path + '/*.png'))

    model = get_rife_model()
    with tqdm(total=len(imgs_paths)) as pbar:
        for s in slices:
            if isinstance(s, tuple):
                in_paths = imgs_paths[slice(*s)]

                infer_rife(in_paths=in_paths, out_path=first_inter, keep_source_imgs=False, model=model, tqdm_bar=pbar)
                in_paths2 = sorted(glob(first_inter + '/*.png'))
                infer_rife(in_paths=in_paths2, out_path=out_path, keep_source_imgs=False, starting_index=s[0] + 1,
                           model=model)

                shutil.rmtree(first_inter, ignore_errors=True)
                os.makedirs(first_inter, exist_ok=True)

                shutil.copy(imgs_paths[s[0]], out_path + str(s[0]).zfill(6) + ".png")
                shutil.copy(imgs_paths[s[1] - 1], out_path + str(s[1] - 1).zfill(6) + ".png")
            else:  # 1 frame
                shutil.copy(imgs_paths[s], out_path)
                pbar.update(1)


if __name__ == '__main__':
    rife_stage()
