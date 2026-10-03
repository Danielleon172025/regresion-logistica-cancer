"""Métricas de clasificación e intervalos de confianza por bootstrap.

La clase positiva es `maligno = 1`. Por eso:
- recall (sensibilidad) = proporción de tumores malignos detectados,
- precisión = de los casos marcados como malignos, cuántos lo eran,
- especificidad = proporción de tumores benignos reconocidos como benignos.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, fbeta_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

import config


def calcular_metricas(y_real, prob, umbral: float = 0.5) -> dict:
    """Todas las métricas del informe para un umbral dado."""
    y_real = np.asarray(y_real)
    pred = (np.asarray(prob) >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, pred, labels=[0, 1]).ravel()
    return {
        "umbral": float(umbral),
        "vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp),
        "exactitud": accuracy_score(y_real, pred),
        "exactitud_balanceada": balanced_accuracy_score(y_real, pred),
        "precision": precision_score(y_real, pred, zero_division=0),
        "recall": recall_score(y_real, pred),
        "especificidad": tn / (tn + fp),
        "f1": f1_score(y_real, pred),
        "f2": fbeta_score(y_real, pred, beta=config.BETA_UMBRAL),
        "mcc": matthews_corrcoef(y_real, pred),
        "auc_roc": roc_auc_score(y_real, prob),
        "auc_pr": average_precision_score(y_real, prob),
        "brier": brier_score_loss(y_real, prob),
    }


def bootstrap_ic(y_real, prob, umbral: float, metricas=("recall", "precision", "auc_roc")) -> dict:
    """Intervalo de confianza percentil por bootstrap sobre el conjunto de prueba.

    Con solo 143 casos de prueba, una sola cifra de recall esconde mucha
    incertidumbre: el intervalo permite leerla con prudencia.
    """
    y_real = np.asarray(y_real)
    prob = np.asarray(prob)
    rng = np.random.default_rng(config.SEMILLA)
    n = len(y_real)
    muestras = {m: [] for m in metricas}
    for _ in range(config.N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_real[idx])) < 2:      # una remuestra con una sola clase no sirve
            continue
        valores = calcular_metricas(y_real[idx], prob[idx], umbral)
        for m in metricas:
            muestras[m].append(valores[m])
    inferior = (1 - config.NIVEL_IC) / 2 * 100
    superior = 100 - inferior
    return {m: (float(np.percentile(v, inferior)), float(np.percentile(v, superior)))
            for m, v in muestras.items()}


def bootstrap_diferencia(y_real, prob, umbral_a: float, umbral_b: float, metrica: str = "recall") -> dict:
    """Diferencia (b menos a) de una métrica entre dos umbrales, con IC por bootstrap pareado.

    Ambos umbrales se evalúan sobre las mismas remuestras. Comparar dos intervalos
    separados no sirve para saber si la diferencia es real, porque ignora que
    salen de los mismos casos.
    """
    y_real = np.asarray(y_real)
    prob = np.asarray(prob)
    rng = np.random.default_rng(config.SEMILLA)
    n = len(y_real)
    diferencias = []
    for _ in range(config.N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_real[idx])) < 2:
            continue
        a = calcular_metricas(y_real[idx], prob[idx], umbral_a)[metrica]
        b = calcular_metricas(y_real[idx], prob[idx], umbral_b)[metrica]
        diferencias.append(b - a)
    inferior = (1 - config.NIVEL_IC) / 2 * 100
    punto = calcular_metricas(y_real, prob, umbral_b)[metrica] - calcular_metricas(y_real, prob, umbral_a)[metrica]
    return {"diferencia": float(punto),
            "ic95": (float(np.percentile(diferencias, inferior)),
                     float(np.percentile(diferencias, 100 - inferior)))}
