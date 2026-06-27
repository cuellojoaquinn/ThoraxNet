# Dataset

## Fuente

**CheXpert Small** — Stanford ML Group  
Disponible en Kaggle: https://www.kaggle.com/datasets/ashery/chexpert

CheXpert es un dataset de radiografías de tórax con etiquetas para 13 patologías, generadas automáticamente a partir de informes radiológicos. La versión *small* contiene imágenes redimensionadas a 320×320 píxeles.

| Split | Imágenes |
|-------|----------|
| Train | ~191 000 |
| Valid | ~200     |

Las etiquetas pueden ser positivas (1), negativas (0) o inciertas (−1). El modelo usa una estrategia mixta U-Zeros / U-Ones por patología para resolver los valores inciertos durante el entrenamiento.

---

## Descarga manual

1. Crear una cuenta en [Kaggle](https://www.kaggle.com) si no tenés una.
2. Ir a https://www.kaggle.com/datasets/ashery/chexpert y descargar el ZIP.
3. Descomprimir. La estructura que obtenés es:

```
CheXpert-v1.0-small/
├── train.csv
├── valid.csv
├── train/
│   ├── patient00001/
│   │   └── study1/
│   │       └── view1_frontal.jpg
│   └── ...
└── valid/
    └── ...
```

4. Mover el contenido descomprimido a `data/chexpert/` dentro del repo:

```
data/
└── chexpert/
    ├── train.csv
    ├── valid.csv
    ├── train/
    └── valid/
```

---

## Script de corrección de paths

El script `utils/fix_chexpert_paths.py` actualiza las rutas en los archivos CSV para que coincidan con el formato que esperan los DataLoaders del proyecto.

CheXpert organiza las imágenes en subcarpetas anidadas por paciente y estudio. El script aplana esas rutas al formato `patientXXXX_studyY_viewZ.jpg`.

```bash
python utils/fix_chexpert_paths.py
```

Modifica `data/chexpert/train.csv` y `data/chexpert/valid.csv` en el lugar.

**Antes:** `CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg`  
**Después:** `chexpert/train/patient00001_study1_view1_frontal.jpg`

> Ejecutar este script es **necesario** antes de usar los DataLoaders del notebook de entrenamiento (`dev/02-model-training-final.ipynb`).
