import numpy as np
import math as m
import PIL.Image as I
import matplotlib.pyplot as plt
import rawpy
import os
import pandas as pd
from skimage.filters import gaussian
from scipy import signal, integrate
from pathlib import Path
from numpy.typing import NDArray
from skimage.filters import rank as skr
from skimage.morphology import disk
from typing import List, Dict
import seaborn as sns

MODE = "DEBUG"


def take_signal(im: NDArray, bbox: List[int], s_minus: NDArray, info, name, i):
    
    roi_im = im[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    
    # med_im = skr.median(roi_im,disk(150))
    # gauss_im = gaussian(roi_im, sigma=(15,roi_im.shape[1]), preserve_range=True)
    # roi_im_post_treated = np.clip((gauss_im / med_im),0,1)
    
    # concatenate_roi_im_mean = np.mean(roi_im_post_treated, axis=1)*255
    concatenate_roi_im_mean = np.mean(roi_im, axis=1)
    signal = (concatenate_roi_im_mean / s_minus)
    cliped_signal = np.clip(signal, 0, 1) * 255
    
    info[f"{name}_roi_im_{i}"] = roi_im
    info[f"{name}_signal_{i}"] = concatenate_roi_im_mean
    info[f"{name}_treated_signal_{i}"] = signal
    info[f"{name}_cliped_signal_{i}"] = cliped_signal
    # info[f"{name}_med_im"] = med_im
    # info[f"{name}_gauss_im"] = gauss_im
    # info[f"{name}_roi_im_post_treated"] = roi_im_post_treated
    
    return cliped_signal
    
    
def treat_signal(sig, dx):
    min_smooth = -np.min(sig)+255
    peaks, props = signal.find_peaks(-sig+255,height=0.9999999*min_smooth, width=np.zeros_like(sig))
    if len(peaks) == 1:
        y = -sig + 255
        t0 = max(1,m.ceil(peaks - props["widths"]))
        t1 = max(2,m.ceil(peaks + props["widths"]))
        print(t0, t1, peaks, props["widths"])
        area = integrate.simps(y[t0:t1+1], dx=dx)
        height = np.max(props['prominences'])
        return (height, area)
    else:
        y = -sig + 255
        print(f'More than one peak is not handled yet. There is {len(peaks)} at {peaks}. \n return (0,0) because a white strip is assumed')
        return (0, 0)


def open_image(path: Path, ext: str):
    if ext == ".jpg" or ext == ".png":
        im = I.open(path)
        im_rgb = np.asarray(im.convert("RGB"))
    elif ext == ".dng":
        path = path.as_posix()
        with rawpy.imread(path) as raw:
                im_rgb = raw.postprocess()
    return im_rgb


def find_region_of_interest(path: Path, output_dir: Path, name: str, ext: str, bbox: List[int], s_minus: NDArray, info: Dict):
    im_rgb = open_image(path, ext)
    
    roi_rgb = im_rgb[bbox[2]:bbox[3], bbox[0]:bbox[1]]

    signals = [take_signal(im_rgb[...,i], bbox, s_minus[i], info, name, i) for i in range(3)]  # R, G, B
    plt.figure(figsize=(8,8))
    plt.subplot(1,3,1)
    plt.imshow(im_rgb[...,0])
    plt.subplot(1,3,2)
    plt.imshow(im_rgb[...,1])
    plt.subplot(1,3,3)
    plt.imshow(im_rgb[...,2])
    plt.tight_layout()
    os.makedirs(output_dir.joinpath(f"{name}"), exist_ok=True)
    plt.savefig(output_dir.joinpath(f"{name}/roi_RGB.png"))
    plt.show()
    plt.figure()
    plt.imshow(roi_rgb)
    plt.show()
    
    return (signals, roi_rgb)


def post_treatment(path, output_dir: Path, name: str, ext: str, info: Dict, bbox: List[int], s_minus: NDArray):
    signals, info[f"{name}_roi_rgb"] = find_region_of_interest(path, output_dir,name,ext, bbox, s_minus, info)
    treated_signals = [treat_signal(signals[i], dx=0.1) for i in range(len(signals))]
    return signals, treated_signals    


def take_signal_to_substract(blank_img: NDArray, bbox: List[int]):
    roi = blank_img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    mean_s = np.mean(roi, axis=1)
    med_s = [signal.medfilt(mean_s[:,i], kernel_size=101) for i in range(3)]
    # ploting plot of mean_s and med_s with 'r' for channel R, 'g' for channel G and 'b' for channel B on one plot
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    plt.plot(mean_s[:,0], 'r')
    plt.plot(mean_s[:,1], 'g')
    plt.plot(mean_s[:,2], 'b')
    plt.xlabel("pixel", fontsize=15)
    plt.ylabel("intensity", fontsize=15)
    plt.title("mean of the ROI", fontsize=15)
    sns.set_style('whitegrid')
    plt.grid(alpha=0.7)
    plt.subplot(1,2,2)
    plt.plot(med_s[0], 'r')
    plt.plot(med_s[1], 'g')
    plt.plot(med_s[2], 'b')
    plt.xlabel("pixel", fontsize=15)
    plt.ylabel("intensity", fontsize=15)
    plt.title("median of the ROI", fontsize=15)
    # adding a layout to have a better view of the plot as horizontal figure
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    sns.set_style('whitegrid')
    plt.grid(alpha=0.7)
    plt.show()


    return (med_s, mean_s, roi)


def main(input_dir: Path, output_dir: Path):
    paths = next(os.walk(input_dir))[-1]
    first_path = input_dir.joinpath(paths[0]) if Path(paths[0]).suffix != '.csv' else input_dir.joinpath(paths[1])
    imgs_names = []
    info = {}
    R, G, B = [[],[]], [[],[]], [[],[]]
    
    im_bbox = open_image(first_path, first_path.suffix)
    
    fig = plt.figure()
    plt.imshow(im_bbox)
    plt.title("Write the 4 coordinates to crop the image, \nclose the image, write them in the terminal")
    plt.tight_layout()
    plt.show()
    plt.close()
    
    bbox = input("Enter x_start x_stop y_start y_stop:").split(" ")
    for i in range(len(bbox)):
        bbox[i] = int(bbox[i])
    signal_to_substract, info["mean_s_blank"], info["roi_blank"] = take_signal_to_substract(im_bbox, bbox)
    for path in paths:
        path = input_dir.joinpath(path)
        ext = path.suffix
        name = path.stem
        if ext == ".csv": continue
        imgs_names.append(name)
        print("Ext OK")
        signals, treated_signals = post_treatment(path, output_dir, name, ext, info, bbox, signal_to_substract)
        R[0].append(treated_signals[0][0])
        R[1].append(treated_signals[0][1])
        G[0].append(treated_signals[1][0])
        G[1].append(treated_signals[1][1])
        B[0].append(treated_signals[2][0])
        B[1].append(treated_signals[2][1])
        print(type(treated_signals[2][1]))
        
        x = [i for i in range(len(info[f"{name}_signal_1"]))]
        plt.figure(figsize=(8,8))
        plt.plot(x, info[f"{name}_signal_1"], label="Original Green signal")
        plt.plot(x, info[f"{name}_treated_signal_1"]*255, label="Calibrated Green Signal")
        plt.plot(x, info[f"{name}_cliped_signal_1"], label="Cliped signal")
        plt.plot(x, signal_to_substract[1], label="Calibration Signal")
        plt.legend()
        plt.title(f"125 ng/ml signal")
        os.makedirs(output_dir.joinpath(f"{name}"), exist_ok=True)
        plt.savefig(output_dir.joinpath(f"{name}/signal_treatment.png"))
        plt.show()
    df = pd.DataFrame({
                        "Name": imgs_names,
                        "R_height": R[0],
                        "R_area": R[1],
                        "G_height": G[0],
                        "G_area": G[1],
                        "B_height": B[0],
                        "B_area": B[1],
                    })

    df.to_csv(input_dir.joinpath("z_data_final.csv"), index=False, float_format='%g', decimal=',')
    print(df)



if __name__=='__main__':
    input_dir = Path(sys.argv[1])
    # output_dir = Path(sys.argv[2])
    # input_dir = Path("D:/NANO/Final/jpg")
    output_dir = input_dir.joinpath("DEBUG")
    os.makedirs(output_dir, exist_ok=True)
    print(input_dir, output_dir)

    main(input_dir, output_dir, concentrations)