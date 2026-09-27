"""Sensor-degradation augmentation (paper Section III-C, second half).

Section III-C describes OOD injection in two parts. The first is CutMix
(data/cutmix.py). The second, implemented here:

    "Furthermore, random batches are subjected to severe visual degradations,
     including Gaussian noise, motion blur, and fog filters, actively
     penalizing the network via L_calib if it maintains high-confidence
     predictions on compromised inputs."

The purpose is NOT to make detection harder for its own sake. It is to
create inputs where high confidence is *objectively wrong*, so that L_calib
has something to push against. A model that stays certain through heavy fog
is miscalibrated by definition, and without degraded samples in the batch
there is nothing in the data teaching it otherwise.

Consequences that shape the design:

  * Degradation is a CAMERA-level corruption. It changes pixels only. The
    segmentation label and the OOD target are untouched -- fog does not move
    the object, and the object is still anomalous underneath it.
  * It is applied AFTER CutMix compositing and BEFORE normalization, because
    that is where a real camera sits in the chain: the sensor degrades the
    whole scene, including anything in it.
  * Each sample reports whether it was degraded, so Phase 2b can weight
    L_calib differently on degraded vs clean inputs.

Severity ranges are deliberately wide. "Severe" is the paper's word, and a
barely-visible corruption gives the calibration loss nothing to work with.
"""

import numpy as np
from scipy import ndimage

import config


def gaussian_noise(image, severity, rng):
    """Sensor noise: low light, high ISO, a failing sensor.

    image: float32 HxWx3 in [0,1]. Returns the same.
    """
    sigma = severity
    noisy = image + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0.0, 1.0)


def motion_blur(image, severity, rng):
    """Camera or subject motion: the vehicle moving, or a bumpy road.

    A line kernel at a random angle, which is what linear motion during the
    exposure window actually produces -- as opposed to a Gaussian blur, which
    looks like defocus and is a different failure.
    """
    length = max(3, int(round(severity)))
    if length % 2 == 0:
        length += 1

    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    angle = rng.uniform(0.0, 180.0)
    kernel = ndimage.rotate(kernel, angle, reshape=False, order=1, mode="constant")
    total = kernel.sum()
    if total <= 1e-6:                      # a rotation can zero the kernel out
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        total = kernel.sum()
    kernel /= total

    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = ndimage.convolve(image[..., c], kernel, mode="reflect")
    return np.clip(out, 0.0, 1.0)


def fog(image, severity, rng):
    """Atmospheric scattering: fog, haze, heavy spray.

    Standard atmospheric-scattering form, I' = I*t + A*(1-t), where A is the
    airlight (bright grey) and t is transmission. t decreases with distance,
    and in a forward-facing road image distance grows toward the horizon --
    so t is ramped vertically rather than applied flat, which is both more
    physically honest and visually far more convincing than a uniform wash.
    """
    h, w = image.shape[:2]
    airlight = rng.uniform(0.75, 0.95)

    # Horizon sits near the vertical middle of a road scene; fog is densest
    # there and thins toward the immediate foreground at the bottom.
    rows = np.linspace(0.0, 1.0, h, dtype=np.float32)
    depth = 1.0 - np.abs(rows - 0.45) / 0.55
    depth = np.clip(depth, 0.15, 1.0)

    transmission = 1.0 - severity * depth[:, None, None]
    transmission = np.clip(transmission, 0.05, 1.0)

    out = image * transmission + airlight * (1.0 - transmission)
    return np.clip(out, 0.0, 1.0)


# name -> (function, (severity_low, severity_high))
DEGRADATIONS = {
    "gaussian_noise": (gaussian_noise, config.DEGRADATION_NOISE_SIGMA),
    "motion_blur": (motion_blur, config.DEGRADATION_BLUR_LENGTH),
    "fog": (fog, config.DEGRADATION_FOG_STRENGTH),
}


def apply_random_degradation(image_uint8, rng=None):
    """Applies one randomly chosen degradation at a random severity.

    image_uint8: HxWx3 uint8. Returns (HxWx3 uint8, name_of_degradation).
    """
    rng = rng or np.random.default_rng()
    name = str(rng.choice(sorted(DEGRADATIONS)))
    fn, (lo, hi) = DEGRADATIONS[name]
    severity = rng.uniform(lo, hi)

    image = image_uint8.astype(np.float32) / 255.0
    out = fn(image, severity, rng)
    return (out * 255.0).astype(np.uint8), name
