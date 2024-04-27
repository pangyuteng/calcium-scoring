import os
import sys
import traceback
import pandas as pd
import tempfile
import nibabel as nib
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_DIR)
from calcium_scoring import score

def process(img_folder):
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            img_nifti_file = os.path.join(tmpdir,'img.nii.gz')
            seg_folder = os.path.join(tmpdir,'segmentations')
            # dicom to nifti    
            dicom_file_list = [os.path.join(img_folder,x) for x in os.listdir(img_folder) if x.endswith('.dcm')]
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_file_list)
            img_obj = reader.Execute()
            sitk.WriteImage(img_obj,img_nifti_file)

            # segment coronary artery
            totalsegmentator(img_nifti_file, seg_folder,task="coronary_arteries")
            mask_nifti_file = os.path.join(seg_folder,"coronary_arteries.nii.gz")
            
            if not os.path.exists(img_nifti_file) or not os.path.exists(mask_nifti_file):
                raise valueError()

            img_obj = sitk.ReadImage(img_nifti_file)
            mask_obj = sitk.ReadImage(mask_nifti_file)
            agatston_score, volume_score, median_hu, mask_volume = score(img_obj,mask_obj)
            return agatston_score, volume_score, None
    except:
        traceback.print_exc()
        return None, None, traceback.format_exc()

def compute_results(root_folder,output_csv_file):
    case_dir_list = os.listdir(root_folder)
    mylist = []
    for case_id in case_dir_list:
        img_folder = os.path.join(root_folder,case_id,case_id)
        if not os.path.exists(img_folder):
            continue
        agatston_score, volume_score, error_msg = process(img_folder)
        myitem = dict(
            case_id=case_id,
            agatston_score=agatston_score,
            volume_score=volume_score,
            error_msg=error_msg,
        )
        mylist.append(myitem)
        print(myitem)
    df = pd.DataFrame(mylist)
    df.to_csv(output_csv_file,index=False)

if __name__ == "__main__":

    root_folder = sys.argv[1]
    output_csv_file = "coca-results.csv"

    if not os.path.exists(output_csv_file):
        compute_results(root_folder,output_csv_file)

    xls_file = os.path.join(root_folder,"scores.xlsx")
    gt_df = pd.read_excel(xls_file)
    df = pd.read_csv(output_csv_file)
    
"""

docker run -it --gpus 'device=0' --ipc=host -v /mnt:/mnt wasserth/totalsegmentator:2.0.0 bash
pip install openpyxl
python test_cac.py \
    /mnt/scratch/data/coca-dataset/cocacoronarycalciumandchestcts-2/deidentified_nongated


"""