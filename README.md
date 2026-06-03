# Clasificación multi-label de patologías en radiografías de tórax

Datasets: **ChestXray8** (NIH) · **CheXpert** (Stanford) · **VinBigData** (Kaggle)  
Tarea: clasificación multi-label con 11 labels canónicos.

---

## Estructura del repositorio

```
.
├── .ipynb                          # Exploración + unificación + corrección de paths
├── dev/
│   └── 01_dataset_preparation.ipynb   # Entrega TP: Dataset, DataLoaders, augmentation
├── data/
│   ├── README.md                   # Schema y fuentes de los CSVs
│   ├── train.csv                   # Split final de entrenamiento
│   ├── val.csv                     # Split de validación (15% de train)
│   └── test.csv                    # Split de test
├── prod/                           # Modelos entrenados (vacío por ahora)
└── .gitignore
```

---

## Orden de ejecución

### Paso 1 — Instalar dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib pillow scikit-learn jupyter
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Paso 2 — Unificación y corrección de paths (`.ipynb`)

Abrir `.ipynb` y ejecutar **todas las celdas en orden**:

| Sección | Celdas | Qué hace |
|---|---|---|
| 1–3 | Exploración | Carga y analiza cada dataset por separado |
| 4 | Mapeo | Define los diccionarios de labels canónicos |
| 5 | Unificación | Genera `unified_dataset.csv` |
| 6 | Train/Test split | Genera `unified_train.csv` y `unified_test.csv` |
| **7** | **Corrección de paths + val split** | **Genera `data/train.csv`, `data/val.csv`, `data/test.csv`** |

> La sección 7 es la más importante: corrige los paths incorrectos de los 3 datasets
> y crea el split de validación (15% de train, a nivel de paciente para CheXpert).

### Paso 3 — Pipeline de datos (`dev/01_dataset_preparation.ipynb`)

Abrir y ejecutar todas las celdas. Requiere que `data/train.csv` ya exista (Paso 2).

| Sección | Qué hace |
|---|---|
| Setup | Importa librerías, verifica paths |
| Labels canónicos | Define las 11 clases |
| (a) Clase Dataset | `ChestXRayDataset` + DataLoaders |
| (b) Particionado | Tabla de distribución por fuente y split |
| (c) Preprocesamiento | Transforms para val/test (224×224 + normalización ImageNet) |
| (d) Augmentation | Transforms para train + visualización de efectos |
| (e) Verificación | Tensores `(32,3,224,224)` y `(32,11)` + grilla de imágenes desnormalizadas |

**Tiempo estimado:** 2–5 minutos (SSD) / 5–10 minutos (HDD).

---

## Datasets — descarga

Los archivos de imagen **no están en el repositorio** (ver `.gitignore`).

| Dataset | Fuente | Destino |
|---|---|---|
| ChestXray8 | [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) | `ChestXray8/images_001/images/` … `images_012/images/` |
| CheXpert | [Stanford](https://stanfordmlgroup.github.io/competitions/chexpert/) | `chexpert/train/` y `chexpert/valid/` |
| VinBigData | [Kaggle](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) | `VinBigData/train/` |

---

## Labels canónicos (11 clases)

`Atelectasis` · `Cardiomegaly` · `Consolidation` · `Edema` · `Pleural_Effusion`  
`Pneumonia` · `Pneumothorax` · `Infiltration` · `Nodule_Mass` · `Pleural_Thickening` · `No_Finding`
