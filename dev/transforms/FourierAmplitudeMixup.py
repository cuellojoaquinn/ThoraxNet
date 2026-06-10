import random

import numpy as np
from PIL import Image


class FourierAmplitudeMixup:
    """Intercambia amplitudes de baja frecuencia en el espacio de Fourier.

    Preserva la fase (estructura anatómica) y mezcla la amplitud (estilo visual)
    con una imagen aleatoria del pool de entrenamiento.

    beta — fracción del tamaño de imagen usada como radio de la ventana de baja freq.
    p    — probabilidad de aplicar la augmentation por imagen.
    seed — semilla para reproducibilidad.

    Ref: Yang & Soatto (2020), FDA: Fourier Domain Adaptation, CVPR 2020.
    """

    def __init__(self, datasets: list, beta=0.01, p=0.5, seed=42):
        self.datasets = datasets
        self.beta     = beta
        self.p        = p
        self._rng     = random.Random(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self._rng.random() > self.p:
            return img

        ds      = self._rng.choice(self.datasets)
        idx     = self._rng.randint(0, len(ds) - 1)
        row     = ds.df.iloc[idx]
        src_img = ds._load_image(row)          # siempre PIL, sin transform
        src_img = src_img.resize(img.size, Image.BILINEAR)

        img_np = np.array(img).astype(np.float32)
        src_np = np.array(src_img).astype(np.float32)

        mixed = np.stack([
            self._mix_channel(img_np[:, :, c], src_np[:, :, c])
            for c in range(3)
        ], axis=2)

        return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))

    def _mix_channel(self, img_c: np.ndarray, src_c: np.ndarray) -> np.ndarray:
        F_img   = np.fft.fft2(img_c)
        F_src   = np.fft.fft2(src_c)
        amp_img = np.fft.fftshift(np.abs(F_img))
        amp_src = np.fft.fftshift(np.abs(F_src))
        phase   = np.angle(F_img)

        h, w   = img_c.shape
        bh, bw = int(h * self.beta), int(w * self.beta)
        ch, cw = h // 2, w // 2

        amp_mix = amp_img.copy()
        amp_mix[ch-bh:ch+bh, cw-bw:cw+bw] = \
            amp_src[ch-bh:ch+bh, cw-bw:cw+bw]

        F_mix = np.fft.ifftshift(amp_mix) * np.exp(1j * phase)
        return np.real(np.fft.ifft2(F_mix))
