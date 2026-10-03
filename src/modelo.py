"""Construcción, selección de hiperparámetros y ajuste del umbral."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     TunedThresholdClassifierCV, cross_val_predict,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

import config


def validacion_cruzada() -> StratifiedKFold:
    """Mismos pliegos en todo el proyecto (estratificados y con semilla)."""
    return StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEMILLA)


def crear_pipeline(C: float = 1.0, class_weight=None) -> Pipeline:
    """Escalado + regresión logística (regularización L2, solver lbfgs).

    - StandardScaler: las variables tienen escalas muy distintas; sin escalar,
      la regularización penalizaría de forma desigual a cada coeficiente.
    - L2 y lbfgs son los valores por defecto de scikit-learn y son adecuados
      para pocas variables y ~400 observaciones.
    - max_iter=1000 evita advertencias de convergencia.
    """
    return Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", LogisticRegression(C=C, class_weight=class_weight,
                                            max_iter=config.MAX_ITER,
                                            random_state=config.SEMILLA)),
    ])


def buscar_C(X_ent, y_ent) -> GridSearchCV:
    """Evalúa cada valor de C por validación cruzada con el AUC (no depende del umbral).

    Se registran también recall y precisión para ver si C cambia algo más que el
    AUC. No se reajusta aquí: el modelo final se construye después con el C que
    elija `elegir_c_una_se`.
    """
    busqueda = GridSearchCV(
        crear_pipeline(),
        param_grid={"clasificador__C": config.VALORES_C},
        scoring={"auc": "roc_auc",
                 "recall": "recall",
                 "precision": make_scorer(precision_score, zero_division=0)},
        refit=False,
        cv=validacion_cruzada(),
        n_jobs=1,
    )
    return busqueda.fit(X_ent, y_ent)


def elegir_c_una_se(resultados_cv: dict) -> float:
    """Regla de un error estándar: el menor C cuyo AUC medio no se aleja del mejor
    más de un error estándar (desviación / raíz del número de pliegos).

    Diferencias de milésimas entre valores grandes de C son ruido; ante ellas se
    prefiere el modelo más regularizado, que es el más estable.
    """
    medias = np.asarray(resultados_cv["mean_test_auc"])
    desviaciones = np.asarray(resultados_cv["std_test_auc"])
    i_mejor = int(medias.argmax())
    error_estandar = desviaciones[i_mejor] / np.sqrt(config.FOLDS)
    admisibles = [c for c, m in zip(config.VALORES_C, medias)
                  if m >= medias[i_mejor] - error_estandar]
    return float(min(admisibles))


def ajustar_umbral(pipeline: Pipeline, X_ent, y_ent) -> TunedThresholdClassifierCV:
    """Busca por validación cruzada el umbral que maximiza F2 (recall > precisión).

    Se usa solo el conjunto de entrenamiento. El conjunto de prueba no interviene
    en ninguna decisión del modelo.
    """
    ajustado = TunedThresholdClassifierCV(
        estimator=clone(pipeline),
        scoring=make_scorer(fbeta_score, beta=config.BETA_UMBRAL),
        cv=validacion_cruzada(),
        refit=True,
        store_cv_results=True,
    )
    return ajustado.fit(X_ent, y_ent)


def probabilidades_oof(pipeline: Pipeline, X_ent, y_ent) -> np.ndarray:
    """Probabilidad de maligno fuera de muestra (cada caso se predice sin haberlo visto)."""
    return cross_val_predict(clone(pipeline), X_ent, y_ent, cv=validacion_cruzada(),
                             method="predict_proba")[:, 1]


def comparar_modelos(X_final, X_candidatas, X30, y, C_optimo: float) -> pd.DataFrame:
    """Compara en validación cruzada la línea base y varias variantes del modelo."""
    scoring = {
        "exactitud": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "auc": "roc_auc",
    }
    n = X_final.shape[1]
    modelos = {
        "Línea base (siempre benigno)": (DummyClassifier(strategy="most_frequent"), X_final),
        f"Regresión logística, {n} variables finales, C={C_optimo:g}":
            (crear_pipeline(C_optimo), X_final),
        f"Regresión logística, {n} variables finales, C={C_optimo:g}, pesos balanceados":
            (crear_pipeline(C_optimo, class_weight="balanced"), X_final),
        f"Regresión logística, {X_candidatas.shape[1]} candidatas (con colinealidad), C={C_optimo:g}":
            (crear_pipeline(C_optimo), X_candidatas),
        f"Regresión logística, {X30.shape[1]} variables, C={C_optimo:g}":
            (crear_pipeline(C_optimo), X30),
    }
    filas = []
    for nombre, (modelo, X) in modelos.items():
        res = cross_validate(modelo, X, y, cv=validacion_cruzada(), scoring=scoring)
        fila = {"modelo": nombre}
        for m in scoring:
            valores = res[f"test_{m}"]
            fila[m] = f"{valores.mean():.3f} ± {valores.std():.3f}"
            fila[f"{m}_media"] = valores.mean()
        filas.append(fila)
    return pd.DataFrame(filas)


def bootstrap_coeficientes(pipeline: Pipeline, X_ent, y_ent) -> np.ndarray:
    """Coeficientes (en escala estandarizada) de modelos reajustados sobre remuestras."""
    rng = np.random.default_rng(config.SEMILLA)
    n = len(X_ent)
    coeficientes = []
    for _ in range(config.N_BOOTSTRAP_COEF):
        idx = rng.integers(0, n, n)
        m = clone(pipeline).fit(X_ent.iloc[idx], y_ent.iloc[idx])
        coeficientes.append(m.named_steps["clasificador"].coef_[0])
    return np.array(coeficientes)
