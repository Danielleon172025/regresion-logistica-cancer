"""Ejecuta el pipeline completo: datos, exploración, modelo, evaluación y figuras.

Uso (desde la raíz del repositorio):
    python src/main.py
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))   # permite `import config`

import config
import datos as dt
import evaluacion as ev
import graficos as gr
import modelo as md


def a_python(obj):
    """Convierte tipos de NumPy para poder guardar el resultado en JSON."""
    if isinstance(obj, dict):
        return {k: a_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> dict:
    config.DIR_DATOS.mkdir(exist_ok=True)
    config.DIR_RESULTADOS.mkdir(exist_ok=True)

    # 1. Datos ---------------------------------------------------------------
    completo = dt.cargar_completo()
    completo.to_csv(config.DIR_DATOS / "cancer_mama_wisconsin.csv", index=False)
    candidatas = dt.seleccionar_candidatas(completo)
    assert candidatas.isna().sum().sum() == 0, "el dataset no debería tener nulos"

    y = candidatas[config.OBJETIVO]
    X_cand = candidatas[config.NOMBRES_CANDIDATAS]
    X30 = completo.drop(columns=config.OBJETIVO)

    # La partición se hace antes de cualquier decisión basada en los datos.
    X_cand_ent, X_cand_pru, y_ent, y_pru = dt.dividir(X_cand, y)
    X30_ent = X30.loc[X_cand_ent.index]

    # 2. Exploración -----------------------------------------------------------
    por_clase = candidatas.groupby(config.OBJETIVO)[config.NOMBRES_CANDIDATAS].agg(["mean", "std"]).round(3)
    por_clase.columns = [f"{v}_{estadistico}" for v, estadistico in por_clase.columns]
    por_clase.to_csv(config.DIR_RESULTADOS / "estadisticos_por_clase.csv")
    gr.distribucion_clases(candidatas)
    gr.histogramas_por_clase(candidatas)
    gr.correlacion(X_cand)
    gr.dispersion(candidatas)

    # 3. Selección de variables por VIF (solo entrenamiento) ----------------------
    finales, historial_vif = dt.eliminar_por_vif(X_cand_ent)
    historial_vif.to_csv(config.DIR_RESULTADOS / "historial_vif.csv", index=False)
    vif_inicial = dt.calcular_vif(X_cand_ent)
    vif_final = dt.calcular_vif(X_cand_ent[finales])
    X_ent, X_pru = X_cand_ent[finales], X_cand_pru[finales]

    # 4. Selección de C (solo con entrenamiento) -------------------------------
    busqueda = md.buscar_C(X_ent, y_ent)
    C_optimo = md.elegir_c_una_se(busqueda.cv_results_)
    tabla_c = pd.DataFrame({
        "C": config.VALORES_C,
        "auc_media": busqueda.cv_results_["mean_test_auc"],
        "auc_desv": busqueda.cv_results_["std_test_auc"],
        "recall_media": busqueda.cv_results_["mean_test_recall"],
        "precision_media": busqueda.cv_results_["mean_test_precision"],
    }).round(4)
    tabla_c.to_csv(config.DIR_RESULTADOS / "seleccion_C.csv", index=False)
    gr.seleccion_c(busqueda.cv_results_, C_optimo)

    # 5. Comparación de variantes (validación cruzada en entrenamiento) --------
    comparacion = md.comparar_modelos(X_ent, X_cand_ent, X30_ent, y_ent, C_optimo)
    comparacion.drop(columns=[c for c in comparacion if c.endswith("_media")]).to_csv(
        config.DIR_RESULTADOS / "comparacion_modelos_cv.csv", index=False)

    # 6. Modelo final y umbral -----------------------------------------------------
    pipeline = md.crear_pipeline(C_optimo)
    ajustado = md.ajustar_umbral(pipeline, X_ent, y_ent)
    umbral = float(ajustado.best_threshold_)
    final = ajustado.estimator_                      # pipeline reajustado con todo el entrenamiento
    prob_oof = md.probabilidades_oof(pipeline, X_ent, y_ent)
    gr.umbral_oof(y_ent, prob_oof, umbral)

    # 7. Evaluación en prueba (se usa una sola vez) ---------------------------------
    prob_pru = final.predict_proba(X_pru)[:, 1]
    prueba_defecto = ev.calcular_metricas(y_pru, prob_pru, 0.5)
    prueba_ajustado = ev.calcular_metricas(y_pru, prob_pru, umbral)
    ic_defecto = ev.bootstrap_ic(y_pru, prob_pru, 0.5)
    ic_ajustado = ev.bootstrap_ic(y_pru, prob_pru, umbral)
    dif_recall = ev.bootstrap_diferencia(y_pru, prob_pru, 0.5, umbral, "recall")
    dif_precision = ev.bootstrap_diferencia(y_pru, prob_pru, 0.5, umbral, "precision")
    gr.matrices_confusion(y_pru, prob_pru, 0.5, umbral)
    gr.roc_y_pr(y_pru, prob_pru)

    # 8. Interpretación de coeficientes ---------------------------------------------
    coef = final.named_steps["clasificador"].coef_[0]
    boot = md.bootstrap_coeficientes(pipeline, X_ent, y_ent)
    tabla_coef = pd.DataFrame({
        "variable": finales,
        "coeficiente": coef,
        "razon_momios": np.exp(coef),
        "ic95_inferior": np.exp(np.percentile(boot, 2.5, axis=0)),
        "ic95_superior": np.exp(np.percentile(boot, 97.5, axis=0)),
        "vif": vif_final.values,
    }).sort_values("coeficiente", ascending=False).round(3)
    tabla_coef.to_csv(config.DIR_RESULTADOS / "coeficientes.csv", index=False)
    gr.odds_ratios(finales, coef, boot)

    # 9. Resumen -------------------------------------------------------------------
    resumen = {
        "entorno": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "datos": {"n": len(candidatas), "n_entrenamiento": len(X_ent), "n_prueba": len(X_pru),
                  "prevalencia_maligno": float(y.mean())},
        "variables_finales": finales,
        "vif_inicial": vif_inicial.round(2).to_dict(),
        "vif_final": vif_final.round(2).to_dict(),
        "C_optimo": C_optimo,
        "n_iter_lbfgs": int(np.max(final.named_steps["clasificador"].n_iter_)),
        "umbral_ajustado": umbral,
        "prueba_umbral_defecto": prueba_defecto,
        "prueba_umbral_ajustado": prueba_ajustado,
        "ic95_umbral_defecto": ic_defecto,
        "ic95_umbral_ajustado": ic_ajustado,
        "diferencia_recall_ajustado_menos_defecto": dif_recall,
        "diferencia_precision_ajustado_menos_defecto": dif_precision,
        "linea_base_exactitud_prueba": float(1 - y_pru.mean()),
    }
    with open(config.DIR_RESULTADOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(a_python(resumen), f, indent=2, ensure_ascii=False)

    print(f"Variables finales: {finales}")
    print(f"C elegido: {C_optimo:g} | umbral ajustado: {umbral:.3f}")
    for nombre, m in [("umbral 0.50", prueba_defecto), (f"umbral {umbral:.2f}", prueba_ajustado)]:
        print(f"[{nombre}] exactitud={m['exactitud']:.3f} precision={m['precision']:.3f} "
              f"recall={m['recall']:.3f} auc={m['auc_roc']:.3f} FN={m['fn']} FP={m['fp']}")
    return resumen


if __name__ == "__main__":
    main()
