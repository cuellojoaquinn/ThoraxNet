from enum import Enum

class ProjectionStrategy(Enum):
    """Dirección de incidencia del rayo X sobre el paciente."""
    PA_ONLY = "PA"   # rayo entra por la espalda — paciente parado (ideal)
    AP_ONLY = "AP"   # rayo entra por el pecho  — paciente acostado
    ALL     = "all"


class ViewStrategy(Enum):
    """Plano de adquisición de la imagen."""
    FRONTAL_ONLY = "Frontal"
    LATERAL_ONLY = "Lateral"
    ALL          = "all"