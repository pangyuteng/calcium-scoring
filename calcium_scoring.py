
"""

Calculate the calcium score via the traditional Agatston scoring technique, as outlined in the 
[original paper](10.1016/0735-1097(90)90282-T). Energy (`kV`) specific `threshold`s are determined based on 
previous [publications](https://doi.org/10.1093/ehjci/jey019). 

#### Inputs
- `vol`: input volume containing just the region of interest
- `spacing`: known pixel/voxel spacing (can be an array of `Unitful.Quantity`)
- `alg::Agatston`: Agatston scoring algorithm `Agatston()`
- kwargs:
  - `kV=120`: energy of the input CT scan image
  - `min_size=1`: minimum connected component size (see [`label_components`](https://github.com/JuliaImages/Images.jl))

#### Returns
- `agatston_score`: total Agatston score
- `volume_score`: total calcium volume via Agatston scoring (can be a `Unitful.Quantity`)


#### References
[Quantification of coronary artery calcium using ultrafast computed tomography](https://doi.org/10.1016/0735-1097(90)90282-t)

[Ultra-low-dose coronary artery calcium scoring using novel scoring thresholds for low tube voltage protocols—a pilot study ](https://doi.org/10.1093/ehjci/jey019)

"""

import sys
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


def score(img_obj,mask_obj,kV=120,min_size=1):
    
    if kV not in [70, 80, 100, 120, 135]:
        raise ValueError("kV value not supported!")

    # resample to 3mm spacing, then mask out top and bottom
    # so we are only using 40 slices.

    og_spacing = list(img_obj.GetSpacing())
    new_spacing = [og_spacing[0],og_spacing[1],3.0]
    img_obj = resample_img(img_obj, out_spacing=new_spacing, is_label=False)
    mask_obj = resample_img(mask_obj, out_spacing=new_spacing, is_label=True)
    
    img = sitk.GetArrayFromImage(img_obj)
    mask = sitk.GetArrayFromImage(mask_obj)
    spacing = list(img_obj.GetSpacing())
    spacing = np.roll(spacing,1) # now z is at 0th index.

    threshold = np.round(378 * np.exp(-0.009 * kV))
    area = spacing[1] * spacing[2]
    min_size_pixels = np.round(min_size / area) # respecting the original implementation.

    agatston_score = 0
    volume_score = 0
    
    vol = img.copy()
    vol[mask==0]=-1000 # ct
    for z in range(vol.shape[0]):
        zslice = vol[z,:,:].squeeze()
        thresholded_slice = zslice > threshold
        max_intensity = np.max(zslice)
        if max_intensity < threshold:
            continue
        label_img = label(thresholded_slice)
        regions = regionprops(label_img)
        #print(z,len(regions))
        #imageio.imwrite(f'{z}.png', (label_img/np.max(label_img)*255).astype(np.uint8) )
        for r in regions:
            region_mask = label_img == r.label
            num_label_idxs = np.sum(region_mask)
            if num_label_idxs < min_size_pixels:
                continue
            intensities = zslice[region_mask==1]
            max_intensity = np.max(intensities)
            weight = _weight_thresholds(kV, max_intensity)
            slice_score = num_label_idxs * area * np.min([weight, 4])
            agatston_score += slice_score
            volume_score += num_label_idxs * np.prod(spacing)

    return agatston_score, volume_score

if __name__ == "__main__":
    img_file = sys.argv[1]
    totalseg_file = sys.argv[2]
    img_obj = sitk.ReadImage(img_file)
    mask_obj = sitk.ReadImage(totalseg_file)
    mask = sitk.GetArrayFromImage(mask_obj)
    mask = np.logical_or(np.logical_or(mask==24,mask==25),mask==26)
    mask = mask.astype(np.int16)
    x,y,z=np.where(mask==1)
    
    print(len(np.unique(x)))

    # locate z location, resample so you get 20 slices?

    mask_obj = sitk.GetImageFromArray(mask)
    mask_obj.SetOrigin(img_obj.GetOrigin())
    mask_obj.SetDirection(img_obj.GetDirection())
    mask_obj.SetSpacing(img_obj.GetSpacing())
    agatston_score, volume_score = score(img_obj,mask_obj)
    print(f"agatston_score: {agatston_score}, volume_score: {volume_score}")


"""

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



python calcium_scoring.py ct.nii.gz segmentations.nii.gz

"""
