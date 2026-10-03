"""Todas las figuras del informe. Cada función guarda un PNG en `figures/`."""
import matplotlib
matplotlib.use("Agg")                     # sin ventanas: guarda directamente a archivo
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_auc_score, average_precision_score, roc_curve)

import config

ROJO = "#c0392b"      # maligno
AZUL = "#2d6cdf"      # benigno
GRIS = "#7f8c8d"


def _guardar(fig, nombre: str) -> None:
    config.DIR_FIGURAS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / nombre, dpi=150)
    plt.close(fig)


def distribucion_clases(datos):
    conteo = datos[config.OBJETIVO].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Benigno", "Maligno"], conteo.values, color=[AZUL, ROJO])
    for i, v in enumerate(conteo.values):
        ax.text(i, v + 5, f"{v} ({v / len(datos):.1%})", ha="center")
    ax.set_ylabel("Número de casos")
    ax.set_title(f"Distribución de las clases (n={len(datos)})")
    ax.set_ylim(0, conteo.max() * 1.15)
    _guardar(fig, "fig01_distribucion_clases.png")


def histogramas_por_clase(datos):
    fig, ejes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, var in zip(ejes.ravel(), config.NOMBRES_CANDIDATAS):
        for valor, color, etiqueta in [(0, AZUL, "benigno"), (1, ROJO, "maligno")]:
            ax.hist(datos.loc[datos[config.OBJETIVO] == valor, var], bins=25,
                    alpha=0.6, color=color, label=etiqueta)
        ax.set_title(var)
        ax.set_ylabel("Casos")
    ejes[0, 0].legend()
    fig.suptitle("Distribución de cada variable según el diagnóstico")
    _guardar(fig, "fig02_histogramas_por_clase.png")


def correlacion(X):
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Correlación de Pearson")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlación entre las variables de entrada")
    _guardar(fig, "fig03_correlacion.png")


def dispersion(datos):
    colores = datos[config.OBJETIVO].map({0: AZUL, 1: ROJO})
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(datos["radio"], datos["concavidad"], c=colores, alpha=0.6, s=18)
    ax.scatter([], [], c=AZUL, label="benigno")
    ax.scatter([], [], c=ROJO, label="maligno")
    ax.set_xlabel("Radio promedio del núcleo")
    ax.set_ylabel("Concavidad (promedio)")
    ax.set_title("Radio y concavidad según el diagnóstico")
    ax.legend()
    _guardar(fig, "fig04_dispersion_radio_concavidad.png")


def seleccion_c(resultados_cv, c_elegido):
    medias = np.asarray(resultados_cv["mean_test_auc"])
    desv = np.asarray(resultados_cv["std_test_auc"])
    C = np.array(config.VALORES_C)
    i_mejor = int(medias.argmax())
    limite = medias[i_mejor] - desv[i_mejor] / np.sqrt(config.FOLDS)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(C, medias, yerr=desv, marker="o", color=AZUL, capsize=3,
                label="AUC (media ± desv. est.)")
    ax.axhline(limite, color=GRIS, linestyle=":", label="Mejor AUC menos 1 error estándar")
    ax.axvline(c_elegido, color="green", linestyle="--", label=f"C elegido = {c_elegido:g}")
    ax.set_xscale("log")
    ax.set_xlabel("C (inverso de la regularización, escala log)")
    ax.set_ylabel("AUC en validación cruzada (5 pliegos)")
    ax.set_title("Selección de C")
    ax.legend(fontsize=8, loc="lower right")
    _guardar(fig, "fig05_seleccion_C.png")


def umbral_oof(y_ent, prob_oof, umbral_elegido):
    from sklearn.metrics import fbeta_score, precision_score, recall_score
    umbrales = np.linspace(0.02, 0.98, 97)
    precision = [precision_score(y_ent, prob_oof >= u, zero_division=0) for u in umbrales]
    recall = [recall_score(y_ent, prob_oof >= u) for u in umbrales]
    f2 = [fbeta_score(y_ent, prob_oof >= u, beta=config.BETA_UMBRAL, zero_division=0)
          for u in umbrales]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(umbrales, recall, color=ROJO, label="Recall (sensibilidad)")
    ax.plot(umbrales, precision, color=AZUL, label="Precisión")
    ax.plot(umbrales, f2, color="black", linestyle=":", label="F2")
    ax.axvline(0.5, color=GRIS, linestyle="--", label="Umbral por defecto (0.50)")
    ax.axvline(umbral_elegido, color="green", linestyle="--",
               label=f"Umbral elegido ({umbral_elegido:.2f})")
    ax.set_xlabel("Umbral sobre la probabilidad de maligno")
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Umbral y métricas con predicciones fuera de muestra (entrenamiento)")
    ax.legend(fontsize=8, loc="lower center")
    _guardar(fig, "fig06_umbral_oof.png")


def matrices_confusion(y_prueba, prob, umbral_defecto, umbral_ajustado):
    fig, ejes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, umbral, titulo in [(ejes[0], umbral_defecto, "Umbral por defecto"),
                               (ejes[1], umbral_ajustado, "Umbral ajustado")]:
        m = confusion_matrix(y_prueba, (prob >= umbral).astype(int), labels=[0, 1])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=16,
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Benigno", "Maligno"])
        ax.set_yticks([0, 1], ["Benigno", "Maligno"])
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title(f"{titulo} ({umbral:.2f})")
    _guardar(fig, "fig07_matrices_confusion.png")


def roc_y_pr(y_prueba, prob):
    fpr, tpr, _ = roc_curve(y_prueba, prob)
    prec, rec, _ = precision_recall_curve(y_prueba, prob)
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.5))
    ejes[0].plot(fpr, tpr, color=AZUL, label=f"AUC = {roc_auc_score(y_prueba, prob):.3f}")
    ejes[0].plot([0, 1], [0, 1], "--", color=GRIS, label="Azar (AUC = 0.5)")
    ejes[0].set_xlabel("Tasa de falsos positivos")
    ejes[0].set_ylabel("Tasa de verdaderos positivos (recall)")
    ejes[0].set_title("Curva ROC (conjunto de prueba)")
    ejes[0].legend()
    ejes[1].plot(rec, prec, color=ROJO,
                 label=f"AP = {average_precision_score(y_prueba, prob):.3f}")
    ejes[1].axhline(np.mean(y_prueba), linestyle="--", color=GRIS,
                    label=f"Azar (prevalencia = {np.mean(y_prueba):.2f})")
    ejes[1].set_xlabel("Recall")
    ejes[1].set_ylabel("Precisión")
    ejes[1].set_title("Curva precisión-recall (conjunto de prueba)")
    ejes[1].legend()
    _guardar(fig, "fig08_roc_pr.png")


def odds_ratios(nombres, coef, coef_bootstrap):
    """Razones de momios por desviación estándar, con IC 95 % por bootstrap."""
    orr = np.exp(coef)
    inf = np.exp(np.percentile(coef_bootstrap, 2.5, axis=0))
    sup = np.exp(np.percentile(coef_bootstrap, 97.5, axis=0))
    orden = np.argsort(orr)
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(nombres))
    ax.errorbar(orr[orden], y, xerr=[orr[orden] - inf[orden], sup[orden] - orr[orden]],
                fmt="o", color=ROJO, capsize=3)
    ax.axvline(1, color=GRIS, linestyle="--")
    ax.set_yticks(y, np.array(nombres)[orden])
    ax.set_xscale("log")
    ax.set_xlabel("Razón de momios por +1 desviación estándar (escala log)")
    ax.set_title("Efecto de cada variable sobre las probabilidades de maligno")
    _guardar(fig, "fig09_razones_de_momios.png")
