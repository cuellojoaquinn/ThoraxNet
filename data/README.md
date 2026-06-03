# Dataset

Este directorio contiene los CSVs de splits para el proyecto de clasificación multi-label de patologías en radiografías de tórax.

## Fuentes de datos

| Dataset | Fuente | Imágenes | Labels |
|---|---|---|---|
| **ChestXray8** | [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) | 112,120 PNG | 14 patologías |
| **CheXpert** | [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) | 223,648 JPG | 14 labels (con incertidumbre) |
| **VinBigData** | [Kaggle VinBigData Chest X-ray](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) | 15,000 JPG | 15 clases (formato detección) |

## Cómo descargar el dataset

### ChestXray8
```bash
# Requiere cuenta NIH / registración
# Link directo: https://nihcc.app.box.com/v/ChestXray-NIHCC
# Descargar los 12 zips (images_001.tar.gz … images_012.tar.gz) y descomprimir en:
# data/raw/ChestXray8/images_001/images/ … images_012/images/
```

### CheXpert
```bash
# Requiere registración en Stanford
# https://stanfordmlgroup.github.io/competitions/chexpert/
# Descomprimir en: data/raw/chexpert/train/ y data/raw/chexpert/valid/
```

### VinBigData
```bash
# Requiere cuenta Kaggle
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection
# Descomprimir en: data/raw/VinBigData/train/ y data/raw/VinBigData/test/
```

## Labels canónicos (11 clases)

`Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Pleural_Effusion`,
`Pneumonia`, `Pneumothorax`, `Infiltration`, `Nodule_Mass`, `Pleural_Thickening`, `No_Finding`

## Schema de los CSVs

```
image_id | source | image_path | Atelectasis | ... | No_Finding
```

- `image_path`: path relativo a la raíz del dataset
- Labels: `1.0` (positivo), `0.0` (negativo), vacío/NaN (no disponible o incierto)

## Particiones

| Split | Archivo | Filas aprox. |
|---|---|---|
| Entrenamiento | `train.csv` | ~266,000 |
| Validación | `val.csv` | ~47,000 |
| Test | `test.csv` | ~26,000 |

- **ChestXray8**: split oficial (train_val_list.txt / test_list.txt)
- **CheXpert**: split oficial (train.csv / valid.csv del dataset original)
- **VinBigData**: split aleatorio 80/20 por image_id (seed=42)
- **Validación**: 15% del conjunto train, split aleatorio por image_id (seed=42)
