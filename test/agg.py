import traceback
import os
import sys
import pathlib
import SimpleITK as sitk
import numpy as np
import pandas as pd

from calcium_scoring import score

def main(results_folder):
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

        agatston_score, volume_score, median_hu = score(img_obj,mask_obj,kV=120,min_size=1)
        myitem = dict(
            img_file=img_file,
            mask_file=mask_file,
            spacing=spacing,
            dist_z=dist_z,
            dist_mm=dist_mm,
            agatston_score=agatston_score,
            volume_score=volume_score,
            median_hu=median_hu,
        )
        mylist.append(myitem)
        print(f"agatston_score: {agatston_score}, volume_score: {volume_score}, median_hu: {median_hu}")
        print(f'spacing {spacing} dist {dist_z} voxels {dist_mm} mm')
        pd.DataFrame(mylist).to_csv("scores.csv",index=False)

    pd.DataFrame(mylist).to_csv("scores.csv",index=False)

if __name__ == "__main__":
    results_folder = sys.argv[1]
    main(results_folder)


"""
/mnt/hd2/data/Totalsegmentator_dataset/s1007/

python agg.py /mnt/hd2/data/Totalsegmentator_dataset

"""