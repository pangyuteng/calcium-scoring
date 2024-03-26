
"""
reference

Calculate the calcium score via the traditional Agatston scoring technique

https://github.com/MolloiLab/CalciumScoring.jl/blob/master/src/agatston.jl


https://doi.org/10.1093/ehjci/jey019
https://doi.org/10.1016/0735-1097(90)90282-t
https://doi.org/10.1093/ehjci/jey019

Ulzheimer, S., & Kalender, W. A. (2003). 
Assessment of calcium scoring performance in cardiac computed tomography. 
European radiology, 13, 484-497.
https://pubmed.ncbi.nlm.nih.gov/12594550/



A score for each region of interest was
  calculated by multiplying the density score and the area . A
  total coronary calcium score was determined by adding up
  each of these scores for all 20 slices
    ...
  To evaluate the limitations of scanning only the proximal coronary arteries,
  58 subjects had studies using 40 rather than 20 slices (120
  versus 60 mm), requiring an additional scan and breath hold
  for the distal segments .

Scoring methods and cutoff values differed widely. 
All the studies used the Agatston scoring method. However, 1 study
used only 20 slices of the 6-mm slices to obtain the score, 
and another used 20 slices of the 3-mm slices, 
whereas the others used 40 slices of the 3-mm slice technique.

https://www.sciencedirect.com/science/article/pii/S0002914999009066
resample to 3mm?

https://ajronline.org/doi/10.2214/ajr.176.5.1761295

https://www.sciencedirect.com/science/article/pii/S0735109702029753#BIB15


"""

import sys
import pandas as pd
import numpy as np
import imageio
import SimpleITK as sitk
from skimage.measure import label, regionprops

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def _weight_thresholds(kV, max_intensity):

    if kV not in [70, 80, 100, 120, 135]:
        raise ValueError("kV value not supported!")
    weight = None
    if kV == 70:
        if max_intensity < 207:
            weight = 0
        elif max_intensity < 318:
            weight = 1
        elif max_intensity < 477:
            weight = 2
        elif max_intensity < 636:
            weight = 3
        elif max_intensity >= 636:
            weight = 4
    elif kV == 80:
        if max_intensity < 177:
            weight = 0
        elif max_intensity < 271:
            weight = 1
        elif max_intensity < 408:
            weight = 2
        elif max_intensity < 544:
            weight = 3
        elif max_intensity >= 544:
            weight = 4
    elif kV == 100:
        if max_intensity < 145:
            weight = 0
        elif max_intensity < 222:
            weight = 1
        elif max_intensity < 334:
            weight = 2
        elif max_intensity < 446:
            weight = 3
        elif max_intensity >= 446:
            weight = 4
    elif kV == 120:
        if max_intensity < 130:
            weight = 0
        elif max_intensity < 199:
            weight = 1
        elif max_intensity < 299:
            weight = 2
        elif max_intensity < 399:
            weight = 3
        elif max_intensity >= 399:
            weight = 4
    elif kV == 135:
        if max_intensity < 119:
            weight = 0
        elif max_intensity < 183:
            weight = 1
        elif max_intensity < 274:
            weight = 2
        elif max_intensity < 364:
            weight = 3
        elif max_intensity >= 364:
            weight = 4
    return weight


def score(img_obj,mask_obj,kV=120,min_size_mm2=1,slice_spacing_mm=3.0,max_slice=20):
    
    if kV not in [70, 80, 100, 120, 135]:
        raise ValueError("kV value not supported!")

    # resample to 3mm spacing, then mask out top and bottom
    # so we are only using 40 slices.

    og_spacing = list(img_obj.GetSpacing())
    new_spacing = [og_spacing[0],og_spacing[1],slice_spacing_mm]
    img_obj = resample_img(img_obj, out_spacing=new_spacing, is_label=False)
    mask_obj = resample_img(mask_obj, out_spacing=new_spacing, is_label=True)
    
    img = sitk.GetArrayFromImage(img_obj)
    mask = sitk.GetArrayFromImage(mask_obj)
    spacing = list(img_obj.GetSpacing())
    spacing = np.roll(spacing,1) # now z is at 0th index.

    threshold = np.round(378 * np.exp(-0.009 * kV))
    pixel_area = spacing[1] * spacing[2] # mm^2
    min_size_pixels = np.round(min_size_mm2 / pixel_area) # respecting the original implementation.

    mylist = []
    volume = img.copy()
    volume[mask==0]=-1000 # ct
    for z in range(volume.shape[0]):
        
        slice_agatston_score = 0
        slice_volume_score = 0

        zslice = volume[z,:,:].squeeze()
        thresholded_slice = zslice > threshold
        max_intensity = np.max(zslice)
        if max_intensity < threshold:
            continue
        label_img = label(thresholded_slice)
        regions = regionprops(label_img)

        for r in regions:
            region_mask = label_img == r.label
            num_label_idxs = np.sum(region_mask)
            if num_label_idxs < min_size_pixels:
                continue
            intensities = zslice[region_mask==1]
            max_intensity = np.max(intensities)
            weight = _weight_thresholds(kV, max_intensity)
            slice_score = num_label_idxs * pixel_area * np.min([weight, 4])
            slice_agatston_score += slice_score
            slice_volume_score += num_label_idxs * np.prod(spacing)
        mylist.append(dict(
            slice_idx=z,
            agatston_score=slice_agatston_score,
            volume_score=slice_volume_score,
        ))

    # NOTE: max_slice was mentioned by several papers, likely set as part of image acquisition.
    # thus we are filtering the slices by higer agatston scores
    if len(mylist) > 0:
        df = pd.DataFrame(mylist)
        df.sort_values(['agatston_score'], axis=0, ascending=False,inplace=True)
        df = df.reset_index()
        if len(df) > max_slice:
            df = df.loc[:max_slice,:]

        agatston_score = df.agatston_score.sum()
        volume_score = df.volume_score.sum()
    else:
        agatston_score = 0
        volume_score = 0

    # NOTE: median_hu can be used to determine if series contains contrast, blood is 40HU
    print(volume[mask==1].shape)
    median_hu = np.median(volume[mask==1])
    mask_volume = np.sum(mask==1) * np.prod(spacing)
    return agatston_score, volume_score, median_hu, mask_volume

if __name__ == "__main__":
    img_file = sys.argv[1]
    totalseg_file = sys.argv[2]
    img_obj = sitk.ReadImage(img_file)
    mask_obj = sitk.ReadImage(totalseg_file)
    mask = sitk.GetArrayFromImage(mask_obj)
    mask = np.logical_or(np.logical_or(mask==24,mask==25),mask==26)
    mask = mask.astype(np.int16)
    x,y,z=np.where(mask==1)
    
    print('slice count',len(np.unique(x)))

    # locate z location, resample so you get 20 slices?

    mask_obj = sitk.GetImageFromArray(mask)
    mask_obj.SetOrigin(img_obj.GetOrigin())
    mask_obj.SetDirection(img_obj.GetDirection())
    mask_obj.SetSpacing(img_obj.GetSpacing())
    agatston_score, volume_score, median_hu, mask_volume = score(img_obj,mask_obj)
    print(f"agatston_score: {agatston_score}, volume_score: {volume_score}, median_hu: {median_hu}, mask_volume {mask_volume}")


"""

python calcium_scoring.py ct.nii.gz segmentations.nii.gz

"""
