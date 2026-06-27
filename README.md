# ThoraxNet — Clasificación de patologías en radiografías de tórax

Aplicación de apoyo a la lectura radiológica que detecta 13 hallazgos en radiografías de tórax frontales mediante una red neuronal convolucional. El modelo asigna una probabilidad a cada hallazgo y aplica un umbral óptimo por patología para indicar si fue detectado o no.

> **Resultado orientativo, no es un diagnóstico.** Confirmar siempre con criterio clínico.

---

## Demo

**App desplegada:** https://thoraxnet-test.streamlit.app/

Accesible sin login ni instalación.

---

## Integrantes

- Joaquin Javier Cuello

---

## Modelo

| Parámetro       | Valor                          |
|-----------------|--------------------------------|
| Arquitectura    | DenseNet-121 (preentrenado ImageNet) |
| Optimizador     | AdamW (lr=1e-4, wd=1e-5)      |
| Función de pérdida | Focal Loss (α=1.0, γ=2.0)  |
| Augmentación    | RandomResizedCrop, HorizontalFlip, ColorJitter |
| Batch size      | 128                            |
| Early stopping  | Patience 5 (sobre val loss)   |
| Scheduler       | ReduceLROnPlateau              |
| Dataset         | CheXpert Small (solo vistas frontales) |
| Split           | 80 / 10 / 10 a nivel de paciente |
| Clases          | 13                             |
| AUC promedio    | 0.793                          |
| F1 promedio     | 0.472                          |

Los umbrales de detección por patología se optimizan sobre la curva precision-recall del set de test y se guardan en `model/v2.0/densenet_focal_moderate_test_metrics.json`.

---

## Dataset

**CheXpert Small** — Stanford ML Group  
Kaggle: https://www.kaggle.com/datasets/ashery/chexpert

Ver instrucciones de descarga y preparación en [`data/README.md`](data/README.md).

---

## Correr la app localmente

### 1. Clonar el repositorio

```bash
git clone https://github.com/cuellojoaquinn/ThoraxNet.git
cd ThoraxNet
```

### 2. Instalar dependencias

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install streamlit torch torchvision pillow
```

### 3. Levantar la app

```bash
cd prod
streamlit run app.py
```

La app queda disponible en `http://localhost:8501`.

---

## Estructura del repositorio

```
.
├── data/
│   ├── README.md              # Instrucciones del dataset
│   └── chexpert/              # Dataset (no incluido en el repo)
├── dev/
│   ├── 01_data_preparation_v3.ipynb   # Preparación del dataset
│   └── 02-model-training-final.ipynb  # Entrenamiento y evaluación del modelo
├── model/
│   └── v2.0/
│       ├── densenet_focal_moderate.pth              # Pesos del modelo
│       └── densenet_focal_moderate_test_metrics.json # Métricas y umbrales
├── prod/
│   ├── app.py                 # Aplicación Streamlit
│   ├── utils.py               # Carga del modelo y predicción
│   └── data/                  # Imágenes de ejemplo por patología
└── utils/
    └── fix_chexpert_paths.py  # Corrección de paths del CSV de CheXpert
```
