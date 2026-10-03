"""Carga, selección de variables y partición de los datos."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import config


def cargar_completo() -> pd.DataFrame:
    """Devuelve las 30 variables originales más la etiqueta `maligno`.

    scikit-learn codifica 0 = maligno y 1 = benigno. Aquí se invierte para que la
    clase positiva sea la que importa clínicamente (maligno = 1) y así recall,
    precisión y AUC se lean directamente sobre los tumores malignos.
    """
    bc = load_breast_cancer(as_frame=True)
    datos = bc.data.copy()
    datos[config.OBJETIVO] = (bc.target == 0).astype(int)
    return datos


def seleccionar_candidatas(datos: pd.DataFrame) -> pd.DataFrame:
    """Se queda con las 6 variables candidatas, renombradas al español."""
    subconjunto = datos[list(config.CANDIDATAS) + [config.OBJETIVO]]
    return subconjunto.rename(columns=config.CANDIDATAS)


def dividir(X: pd.DataFrame, y: pd.Series):
    """Partición entrenamiento/prueba estratificada y reproducible."""
    return train_test_split(
        X, y,
        test_size=config.TAMANO_PRUEBA,
        random_state=config.SEMILLA,
        stratify=y,
    )


def calcular_vif(X: pd.DataFrame) -> pd.Series:
    """Factor de inflación de la varianza a partir de la matriz de correlación.

    VIF_j es el elemento j de la diagonal de la inversa de la matriz de
    correlación. Valores por debajo de 5 indican poca multicolinealidad.
    """
    correlacion = np.corrcoef(X.values, rowvar=False)
    return pd.Series(np.diag(np.linalg.inv(correlacion)), index=X.columns, name="VIF")


def eliminar_por_vif(X: pd.DataFrame, umbral: float = config.UMBRAL_VIF):
    """Elimina, de una en una, la variable con mayor VIF hasta que todas queden bajo el umbral.

    Devuelve la lista de variables que se conservan y el historial de pasos.
    Se calcula solo con el conjunto de entrenamiento.
    """
    variables = list(X.columns)
    historial = []
    while True:
        vif = calcular_vif(X[variables])
        peor = vif.idxmax()
        historial.append({"variables": ", ".join(variables), "mayor_vif": peor,
                          "valor": round(float(vif[peor]), 2)})
        if vif[peor] <= umbral:
            return variables, pd.DataFrame(historial)
        variables.remove(peor)
