"""Parámetros globales del proyecto.

Todo lo que pueda cambiar un resultado (semilla, tamaño del conjunto de prueba,
rejilla de hiperparámetros, número de repeticiones bootstrap) se define aquí,
para que el resto de módulos no tenga valores "escondidos".
"""
from pathlib import Path

# ---------- Rutas (relativas a la raíz del repositorio) ----------
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "data"
DIR_FIGURAS = RAIZ / "figures"
DIR_RESULTADOS = RAIZ / "resultados"

# ---------- Reproducibilidad ----------
SEMILLA = 42

# ---------- Datos ----------
# Variables candidatas (promedios del núcleo celular) y su nombre en español.
# Se descartan de entrada "mean perimeter" y "mean area" porque duplican el radio
# (correlación de 0.99 con él). Entre las 6 candidatas restantes, las finales se
# eligen con un criterio explícito: eliminación iterativa por VIF (ver datos.py).
CANDIDATAS = {
    "mean radius": "radio",
    "mean texture": "textura",
    "mean smoothness": "suavidad",
    "mean concavity": "concavidad",
    "mean concave points": "puntos_concavos",
    "mean symmetry": "simetria",
}
NOMBRES_CANDIDATAS = list(CANDIDATAS.values())
UMBRAL_VIF = 5.0                # se elimina la variable con mayor VIF mientras supere este valor
OBJETIVO = "maligno"            # 1 = maligno (clase positiva), 0 = benigno
TAMANO_PRUEBA = 0.25

# ---------- Modelo ----------
FOLDS = 5                                              # validación cruzada estratificada
VALORES_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  # inverso de la regularización L2
MAX_ITER = 1000                                        # holgado: lbfgs converge antes
BETA_UMBRAL = 2                                        # F2: el recall pesa más que la precisión

# ---------- Incertidumbre ----------
N_BOOTSTRAP = 1000
N_BOOTSTRAP_COEF = 500
NIVEL_IC = 0.95
