
#https://github.com/miccunifi/ARNIQA/blob/main/utils/utils_data.py
#from vca_utils.distortions import *
from degmani.degmani_utils.arniqua_distortion import *

distortion_groups = {
    "blur": ["gaublur", "lensblur", "motionblur"],
    "color_distortion": ["colordiff", "colorshift", "colorsat1", "colorsat2"],
    "jpeg": ["jpeg2000", "jpeg"],
    "noise": ["whitenoise", "whitenoiseCC", "impulsenoise", "multnoise"],
    "brightness_change": ["brighten", "darken", "meanshift"],
    "spatial_distortion": ["jitter", "noneccpatch", "pixelate", "quantization", "colorblock"],
    "sharpness_contrast": ["highsharpen", "lincontrchange", "nonlincontrchange"],
}

distortion_groups_mapping = {
    "gaublur": "blur",
    "lensblur": "blur",
    "motionblur": "blur",
    "colordiff": "color_distortion",
    "colorshift": "color_distortion",
    "colorsat1": "color_distortion",
    "colorsat2": "color_distortion",
    "jpeg2000": "jpeg",
    "jpeg": "jpeg",
    "whitenoise": "noise",
    "whitenoiseCC": "noise",
    "impulsenoise": "noise",
    "multnoise": "noise",
    "brighten": "brightness_change",
    "darken": "brightness_change",
    "meanshift": "brightness_change",
    "jitter": "spatial_distortion",
    "noneccpatch": "spatial_distortion",
    "pixelate": "spatial_distortion",
    "quantization": "spatial_distortion",
    "colorblock": "spatial_distortion",
    "highsharpen": "sharpness_contrast",
    "lincontrchange": "sharpness_contrast",
    "nonlincontrchange": "sharpness_contrast",
}

distortion_range = {
    "gaublur": [0.1, 0.5, 1, 2, 5],
    "lensblur": [1, 2, 4, 6, 8],
    "motionblur": [1, 2, 4, 6, 10],
    "colordiff": [1, 3, 6, 8, 12],
    "colorshift": [1, 3, 6, 8, 12],
    "colorsat1": [0.4, 0.2, 0.1, 0, -0.4],
    "colorsat2": [1, 2, 3, 6, 9],
    "jpeg2000": [16, 32, 45, 120, 170],
    "jpeg": [43, 36, 24, 7, 4],
    "whitenoise": [0.001, 0.002, 0.003, 0.005, 0.01],
    "whitenoiseCC": [0.0001, 0.0005, 0.001, 0.002, 0.003],
    "impulsenoise": [0.001, 0.005, 0.01, 0.02, 0.03],
    "multnoise": [0.001, 0.005, 0.01, 0.02, 0.05],
    "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
    "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
    "meanshift": [0, 0.08, -0.08, 0.15, -0.15],
    "jitter": [0.05, 0.1, 0.2, 0.5, 1],
    "noneccpatch": [20, 40, 60, 80, 100],
    "pixelate": [0.01, 0.05, 0.1, 0.2, 0.5],
    "quantization": [20, 16, 13, 10, 7],
    "colorblock": [2, 4, 6, 8, 10],
    "highsharpen": [1, 2, 3, 6, 12],
    "lincontrchange": [0., 0.15, -0.4, 0.3, -0.6],
    "nonlincontrchange": [0.4, 0.3, 0.2, 0.1, 0.05],
}

# distortion_functions = {
#     "gaublur": gaussian_blur,
#     "lensblur": lens_blur,
#     "motionblur": motion_blur,
#     "colordiff": color_diffusion,
#     "colorshift": color_shift,
#     "colorsat1": color_saturation1,
#     "colorsat2": color_saturation2,
#     "jpeg2000": jpeg2000,
#     "jpeg": jpeg,
#     "whitenoise": white_noise,
#     "whitenoiseCC": white_noise_cc,
#     "impulsenoise": impulse_noise,
#     "multnoise": multiplicative_noise,
#     "brighten": brighten,
#     "darken": darken,
#     "meanshift": mean_shift,
#     "jitter": jitter,
#     "noneccpatch": non_eccentricity_patch,
#     "pixelate": pixelate,
#     "quantization": quantization,
#     "colorblock": color_block,
#     "highsharpen": high_sharpen,
#     "lincontrchange": linear_contrast_change,
#     "nonlincontrchange": non_linear_contrast_change,
# }

#https://database.mmsp-kn.de/kadid-10k-database.html
"""
dataset with different distortion levels 
"""


def distortion_functions():
    return None