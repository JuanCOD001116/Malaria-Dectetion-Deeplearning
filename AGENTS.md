# Plan de implementación — Proyecto de detección de malaria con CNN contrastiva + modelos clásicos

## 1) Decisión metodológica central

La idea correcta para este proyecto es **separar claramente dos etapas**:

1. **Aprendizaje de representaciones** con una **CNN entrenada con contrastive learning**.
   - La red **no se usa como clasificador final**.
   - Su salida será un vector de **1024 características** por imagen.
   - Estas características deben ser el **único input** para todos los modelos comparativos.

2. **Comparación justa de modelos** sobre el mismo espacio de embeddings.
   - Regresión logística o LDA/QDA: modelo paramétrico.
   - KNN o Naive Bayes: modelo no paramétrico.
   - Random Forest / Extra Trees / XGBoost: ensamble de árboles.
   - MLP: red neuronal artificial.
   - SVM: máquina de soporte vectorial.

Así se resuelve la observación del profesor:
- no habrá “CNN para clasificar” y luego otros modelos con features distintos;
- la CNN se entrenará **para similitud**, no para clasificación;
- todos los modelos usarán exactamente la **misma representación fija**.

---

## 2) Cómo encaja con la guía

### Sección 2 — Descripción del problema
Debes incluir:
- contexto clínico de la malaria;
- descripción del dataset;
- composición de la base de datos;
- variables de entrada y salida;
- ausencia/presencia de datos faltantes;
- paradigma de aprendizaje elegido.

**Decisión recomendada**:
- paradigma principal: **aprendizaje supervisado en la capa final**, pero con **preentrenamiento auto-supervisado/contrastivo** para obtener embeddings.
- en el reporte conviene decir que el problema se aborda como una tarea de **clasificación binaria sobre representaciones aprendidas automáticamente**.

### Sección 3 — Estado del arte
Debes resumir mínimo 4 trabajos similares y explicar:
- qué paradigma usaron;
- qué técnica usaron;
- cómo validaron;
- qué métricas reportaron;
- cuáles fueron sus resultados.

### Sección 4 — Entrenamiento y evaluación
Aquí van:
- metodología de validación;
- hiperparámetros de cada modelo;
- entrenamiento y resultados;
- métricas con intervalos de confianza;
- train/val/test.

**Mínimo 5 modelos** y deben incluir:
- uno paramétrico,
- uno no paramétrico,
- uno basado en ensamble,
- una red neuronal artificial,
- una SVM.

### Sección 5 — Reducción de dimensión
La forma más limpia de cumplirla es tratar los **1024 embeddings** como el espacio original de variables.

Entonces:
- 5.1: analizar variables/embeddings con correlación, varianza, importancia o medidas discriminativas;
- 5.2: PCA sobre los embeddings;
- 5.3: UMAP sobre los embeddings;
- reevaluar los **2 mejores modelos** sobre las nuevas representaciones.

### Sección 6 — Evaluación
Debes cerrar con:
- discusión de resultados;
- comparación con el estado del arte;
- conclusiones;
- limitaciones y trabajo futuro.

---

## 3) Pipeline técnico recomendado

### Fase A. Preparación del proyecto
1. Descargar y organizar el dataset.
2. Verificar integridad de imágenes.
3. Crear un split estratificado y fijo:
   - train,
   - validation,
   - test.
4. Fijar semilla global y registrar versiones.

### Fase B. EDA
Hacer EDA antes de entrenar:
- número de imágenes por clase;
- tamaño/resolución de las imágenes;
- ejemplos visuales de cada clase;
- histogramas de intensidad;
- brillo/contraste;
- detección de imágenes corruptas;
- detección de duplicados o casi duplicados;
- revisión de variabilidad morfológica;
- ejemplos de augmentations.

### Fase C. Aprendizaje contrastivo
Entrenar una CNN para similitud:
- backbone sugerido: **ResNet18/ResNet34 customizada** o una CNN propia;
- salida del encoder: **1024 dimensiones**;
- proyección contrastiva adicional: MLP pequeño para la pérdida contrastiva;
- pérdida: **NT-Xent / InfoNCE**;
- similitud: **coseno**;
- augmentations: recorte, rotación leve, volteo, jitter suave, blur suave, ruido leve.

### Fase D. Extracción de embeddings
Con el encoder congelado:
- generar embeddings de 1024 dimensiones para train/val/test;
- guardar en `.npy`, `.csv` o `.parquet`;
- usar exactamente esos embeddings para todos los modelos clásicos.

### Fase E. Modelos comparativos
Entrenar y validar:
- Logistic Regression;
- KNN o Naive Bayes;
- Random Forest / Extra Trees;
- MLP;
- SVM.

### Fase F. Dimensionality reduction
Sobre los embeddings:
- correlación / varianza / ranking individual;
- PCA;
- UMAP;
- reevaluar top 2 modelos.

### Fase G. Análisis de similitud
Agregar evidencia visual de que el encoder aprende estructura:
- UMAP 2D coloreado por clase;
- matriz de similitud coseno;
- nearest-neighbors por consulta;
- ejemplo de pares “similares” y “no similares”.

---

## 4) Qué reportar en el artículo

### EDA
- tabla con composición del dataset;
- figuras de ejemplos;
- histogramas y análisis visual.

### Modelo contrastivo
- arquitectura;
- fórmula de la pérdida;
- augmentations;
- hiperparámetros;
- curva de entrenamiento de la pérdida contrastiva;
- ejemplos de similitud/neighborhood.

### Modelos clásicos
- tabla comparativa de hiperparámetros;
- tabla de métricas en train/val/test;
- intervalo de confianza;
- matriz de confusión;
- ROC-AUC;
- F1, precision, recall, balanced accuracy.

### Reducción de dimensión
- porcentaje de reducción;
- impacto en desempeño;
- comparación contra embeddings originales.

### Discusión final
- qué modelo ganó y por qué;
- si PCA/UMAP degradan o preservan información;
- si la representación contrastiva fue útil;
- costo computacional;
- limitaciones del enfoque.

---

## 5) Decisiones para garantizar comparación justa

Estas reglas deben aparecer explícitas en el reporte y en el código:

1. **Todos los modelos consumen la misma matriz de embeddings**.
2. **La CNN contrastiva no se entrena para clasificar**.
3. **No mezclar features manuales con embeddings aprendidos**.
4. **No usar preentrenados en unos modelos y entrenar desde cero en otros**.
5. **Usar el mismo split estratificado para todos**.
6. **No usar oversampling/undersampling si la distribución es balanceada**.
7. **Misma métrica, mismo protocolo, misma semilla**.
8. **Reportar train/val/test y no solo test**.

---

## 6) Estructura recomendada del repositorio

```text
repo/
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── contrastive.yaml
│   ├── classical_models.yaml
│   └── reduction.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_contrastive_training.ipynb
│   ├── 03_classical_models.ipynb
│   └── 04_reduction_and_similarity.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── tests/
├── scripts/
│   ├── prepare_data.py
│   ├── train_contrastive.py
│   ├── extract_embeddings.py
│   ├── train_classical_models.py
│   ├── run_reduction.py
│   └── run_evaluation.py
└── reports/
    └── ieee/
```

---

## 7) Reglas de oro para el código

- reproducible;
- con semilla fija;
- con logging de experimentos;
- con guardado de checkpoints;
- con métricas centralizadas;
- con tests automáticos;
- con validación de shapes;
- con funciones pequeñas y claras;
- sin hardcodear rutas.

---

## 8) Orden de ejecución sugerido

1. EDA.
2. Limpieza y split.
3. Preentrenamiento contrastivo.
4. Extracción de embeddings.
5. Modelos clásicos.
6. PCA/UMAP.
7. Reentrenamiento de top 2.
8. Análisis de similitud.
9. Redacción final del reporte.
