# Regresión logística: diagnóstico de tumores de mama (maligno o benigno)

Autor: Daniel Felipe León Higuera · Curso: Introducción a Machine Learning, Semana 6 · Universidad de Cundinamarca

Ejercicio aplicado de regresión logística. Clasifica 569 tumores de mama reales como malignos o benignos a partir de medidas del núcleo celular, y se centra en las decisiones de modelado: descartar variables colineales, justificar los hiperparámetros, elegir el umbral según el costo de cada error y medir la incertidumbre de los resultados.

| Entregable | Archivo |
|---|---|
| Notebook de Google Colab (ejecutado, con salidas) | [`notebooks/RegresionLogistica_CancerMama.ipynb`](notebooks/RegresionLogistica_CancerMama.ipynb) |
| Informe escrito (PDF) | [`Informe_RegresionLogistica_LeonHigueraDanielFelipe.pdf`](Informe_RegresionLogistica_LeonHigueraDanielFelipe.pdf) |
| Scripts del pipeline | [`src/`](src/) |

Abrir el notebook en Colab: https://colab.research.google.com/github/Danielleon172025/regresion-logistica-cancer/blob/main/notebooks/RegresionLogistica_CancerMama.ipynb

> Aviso: es un ejercicio académico. El modelo no está validado para uso clínico.

## Resumen del problema

- Objetivo (binario): `maligno` (1 = maligno, 0 = benigno). La clase positiva es la de interés clínico, así que el *recall* mide la proporción de tumores malignos detectados.
- Datos: *Breast Cancer Wisconsin (Diagnostic)*, 569 muestras (212 malignas, 37,3 %), incluido en scikit-learn (`load_breast_cancer`). Publicado en el repositorio UCI por Wolberg, Street y Mangasarian (1995). No requiere descarga.
- Variables: seis candidatas (radio, textura, suavidad, concavidad, puntos cóncavos y simetría, promedios del núcleo). Se descartaron de entrada perímetro y área porque duplican el radio.

## Decisiones de modelado

| Decisión | Qué se hizo | Por qué |
|---|---|---|
| Partición | 75 % entrenamiento / 25 % prueba, estratificada, semilla 42, antes de cualquier decisión | El conjunto de prueba se usa una sola vez |
| Variables | Eliminación iterativa por VIF (umbral 5) sobre el entrenamiento: sale `puntos_concavos` (VIF 22,6). Quedan 5 variables | Con colinealidad los coeficientes no se pueden leer por separado |
| Escalado | `StandardScaler` dentro de un `Pipeline` | Escalas muy distintas; evita filtración de datos |
| Modelo | `LogisticRegression` (L2, `lbfgs`, `max_iter=1000`) | Valores por defecto adecuados; converge en 8 iteraciones |
| `C` | 0,1, por validación cruzada de 5 pliegos con AUC y la regla de un error estándar | Diferencias de milésimas entre valores grandes de C son ruido; se prefiere el modelo más regularizado |
| Umbral | 0,285, con `TunedThresholdClassifierCV` maximizando F2, solo con el entrenamiento | Un falso negativo es más grave que un falso positivo |
| Incertidumbre | Bootstrap (1000 remuestras) para métricas y bootstrap pareado entre umbrales; 500 remuestras para las razones de momios | 143 casos de prueba dan cifras poco estables |

## Resultados principales (conjunto de prueba, n = 143)

| Métrica | Umbral 0,50 | Umbral ajustado (0,285) |
|---|---|---|
| Exactitud | 0,937 | 0,923 |
| Recall (sensibilidad) | 0,868 (IC 95 %: 0,771 a 0,945) | 0,962 (IC 95 %: 0,902 a 1,000) |
| Precisión | 0,958 | 0,850 |
| Especificidad | 0,978 | 0,900 |
| Falsos negativos / falsos positivos | 7 / 2 | 2 / 9 |
| AUC-ROC | 0,987 (IC 95 %: 0,973 a 0,997) | 0,987 |

Con el umbral por defecto el modelo deja pasar 7 de 53 tumores malignos; con el umbral ajustado deja pasar 2, a cambio de 7 falsas alarmas más. El aumento del recall (+0,094; IC 95 %: 0,033 a 0,177) se mantiene en un bootstrap pareado. En validación cruzada el modelo con las 30 variables originales rinde mejor (AUC 0,992 frente a 0,980); las cinco variables se eligieron por interpretabilidad. El análisis completo está en el informe.

## Estructura del proyecto

```
regresion-logistica-cancer/
├── README.md
├── requirements.txt
├── Informe_RegresionLogistica_LeonHigueraDanielFelipe.pdf   informe escrito
├── notebooks/
│   └── RegresionLogistica_CancerMama.ipynb    notebook de Colab ejecutado
└── src/
    ├── config.py        parámetros: semilla, rejilla de C, número de remuestras, rutas
    ├── datos.py         carga, recodificación de la etiqueta, partición, VIF
    ├── modelo.py        pipeline, búsqueda de C, umbral, comparación de variantes
    ├── evaluacion.py    métricas, intervalos bootstrap, diferencia pareada
    ├── graficos.py      las nueve figuras
    └── main.py          ejecuta todo el pipeline
```

## Instalación y ejecución

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
```

Activar el entorno (Windows PowerShell: `.venv\Scripts\Activate.ps1`; Git Bash: `source .venv/Scripts/activate`; Linux o Mac: `source .venv/bin/activate`) y luego:

```bash
pip install -r requirements.txt
python src/main.py          # ejecuta el pipeline; escribe resultados/ y figures/ en local
```

El notebook se ejecuta en Google Colab de principio a fin (no necesita instalar nada).

## Reproducibilidad

- Semilla fija (`SEMILLA = 42` en `src/config.py` y en el notebook) para la partición, la validación cruzada y los bootstrap.
- El notebook y los scripts implementan los mismos pasos y producen las mismas cifras.
- Los resultados de este repositorio se generaron con Python 3.14.3, NumPy 2.4.4, pandas 3.0.2 y scikit-learn 1.8.0. `TunedThresholdClassifierCV` exige scikit-learn 1.5 o superior.

## Control de versiones

El proyecto se gestiona con Git. El historial de cambios se consulta con `git log`.

## Referencias (resumen)

Todas de 2024 en adelante, en español y de organizaciones o revistas reconocidas: Organización Mundial de la Salud (2026), Cuenta de Alto Costo (2024), Lee (2025, IBM), Romero Ibarra (2025, *Serie Científica de la UCI*) y seis páginas del Curso intensivo de aprendizaje automático de Google for Developers (2025 y 2026). La lista completa con fechas y enlaces está en el informe.

## Datos y licencia

El dataset proviene del UCI Machine Learning Repository. Las condiciones de uso están en la ficha del dataset.
