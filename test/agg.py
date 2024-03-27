import traceback
import os
import sys
import pathlib
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(THIS_DIR))
from calcium_scoring import score

def gen_csv(results_folder,csv_file):
    file_list = [str(x) for x in pathlib.Path(results_folder).rglob("*segmentations.nii.gz")]
    print(len(file_list))
    mylist = []
    for mask_file in file_list:
        img_file = os.path.join(os.path.dirname(mask_file),'ct.nii.gz')
        if not os.path.exists(mask_file) or not os.path.exists(img_file):
            continue

        try:
            img_obj = sitk.ReadImage(img_file)
        except:
            traceback.print_exc()
            continue

        try:
            mask_obj = sitk.ReadImage(mask_file)
        except:
            traceback.print_exc()
            continue

        spacing = list(mask_obj.GetSpacing())
        mask = sitk.GetArrayFromImage(mask_obj)
        heart = np.logical_or(np.logical_or(mask==24,mask==25),mask==26)
        x,y,z = np.where(heart==1)
        if np.sum(heart) == 0:
            continue
        dist_z = np.max(x)-np.min(x)
        dist_mm = dist_z*spacing[2]

        
        heart = heart.astype(np.int16)
        mask_obj = sitk.GetImageFromArray(heart)
        mask_obj.SetOrigin(img_obj.GetOrigin())
        mask_obj.SetDirection(img_obj.GetDirection())
        mask_obj.SetSpacing(img_obj.GetSpacing())

        agatston_score, volume_score, median_hu, mask_volume = score(img_obj,mask_obj,kV=120)
        myitem = dict(
            img_file=img_file,
            mask_file=mask_file,
            spacing=spacing,
            dist_z=dist_z,
            dist_mm=dist_mm,
            agatston_score=agatston_score,
            volume_score=volume_score,
            median_hu=median_hu,
            mask_volume=mask_volume,
        )
        mylist.append(myitem)
        print(f"agatston_score: {agatston_score}, volume_score: {volume_score}, median_hu: {median_hu}, mask_volume {mask_volume}")
        print(f'spacing {spacing} dist {dist_z} voxels {dist_mm} mm')
        pd.DataFrame(mylist).to_csv(csv_file,index=False)

    pd.DataFrame(mylist).to_csv(csv_file,index=False)

def main(results_folder):
    csv_file = "scores.csv"
    if not os.path.exists(csv_file):
        gen_csv(results_folder,csv_file)
    df = pd.read_csv(csv_file)
    df = df[(df.median_hu<60)&(df.mask_volume>100000)] # ??
    plt.hist(df.agatston_score,bins=np.arange(0,10000,1000))
    plt.grid(True)
    plt.savefig('agatston_score.png')
    

if __name__ == "__main__":

    results_folder = sys.argv[1]
    main(results_folder)


"""
/mnt/hd2/data/Totalsegmentator_dataset/s1007/

python agg.py /mnt/hd2/data/Totalsegmentator_dataset

"""