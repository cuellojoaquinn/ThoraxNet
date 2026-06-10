# datasets/__init__.py
from .CheXpertDataset  import CheXpertDataset
from .ChestXray8Dataset    import ChestXray8Dataset
from .VinBigDataset    import VinBigDataset
from .UnifiedDataset   import UnifiedDataset, CANONICAL_LABELS
from .acquisition import ProjectionStrategy, ViewStrategy