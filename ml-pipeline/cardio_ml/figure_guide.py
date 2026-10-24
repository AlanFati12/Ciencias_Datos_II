"""Plain-language reading guide for every figure the pipeline produces.

Each figure slug has three short texts, in Spanish and English:
    title - what the figure is, in everyday words;
    what  - what it shows and which equation or method is behind it;
    read  - how to read it (what to look at first).
The RunTracker copies the guide into run_manifest.json and TRACEABILITY.md, and
the web app shows it next to each figure, so a reader who is not a data
scientist can interpret every graph. Keys prefixed with the tier ("ecg:...")
override the generic text for that tier.
"""
from __future__ import annotations

G = dict  # short alias: G(es=..., en=...)

GUIDE: dict[str, dict[str, dict[str, str]]] = {
    # ------------------------------------------------------------- data / ETL
    "raw_outliers": {
        "title": G(es="Datos crudos: mediciones imposibles", en="Raw data: impossible measurements"),
        "what": G(es="Diagramas de caja de presión, estatura y peso tal como vienen en el archivo original, antes de limpiar.",
                  en="Boxplots of blood pressure, height and weight exactly as they come in the source file, before cleaning."),
        "read": G(es="Los puntos lejos de la caja son valores imposibles (p. ej. presiones de miles de mmHg o negativas). "
                     "Por eso se aplican los filtros de rangos fisiológicos.",
                  en="Points far from the box are impossible values (e.g. pressures of thousands of mmHg or negative). "
                     "They justify the physiological range filters.")},
    "cleaning_funnel": {
        "title": G(es="Trazabilidad: filas que quedan en cada paso", en="Traceability: rows left after each step"),
        "what": G(es="Cada barra es el conjunto de datos después de un paso de limpieza; entre paréntesis, cuántas filas quitó ese paso.",
                  en="Each bar is the dataset after one cleaning step; in brackets, how many rows that step removed."),
        "read": G(es="Léela de arriba hacia abajo. Ninguna fila desaparece sin explicación: el motivo de cada paso está en TRACEABILITY.md.",
                  en="Read it top to bottom. No row disappears without a reason: each step is explained in TRACEABILITY.md.")},
    "class_balance": {
        "title": G(es="Balance de clases antes y después", en="Class balance before and after"),
        "what": G(es="Cuántos casos hay de cada clase. Se toman todos los de la clase menos frecuente y la misma cantidad, al azar, "
                     "de la más frecuente (submuestreo).",
                  en="How many cases of each class there are. All cases of the rarer class are kept and the same number is "
                     "drawn at random from the common one (undersampling)."),
        "read": G(es="A la derecha las dos barras deben medir igual (50/50). Así la exactitud no se infla respondiendo siempre la clase común.",
                  en="On the right both bars must be equal (50/50). Accuracy cannot then be inflated by always answering the common class.")},
    "distributions": {
        "title": G(es="Distribución de cada variable por clase", en="Distribution of each variable by class"),
        "what": G(es="Histograma con su curva suavizada (KDE) para cada variable, un color por clase. Arriba, la asimetría (skewness).",
                  en="Histogram with its smoothed curve (KDE) for each variable, one colour per class. Above, the skewness."),
        "read": G(es="Si las dos curvas están separadas, esa variable ayuda a distinguir las clases; si se superponen, aporta poco.",
                  en="If the two curves are apart, the variable helps to tell the classes apart; if they overlap, it adds little.")},
    "iqr_boxplots": {
        "title": G(es="Valores atípicos con la regla IQR", en="Outliers with the IQR rule"),
        "what": G(es="Cajas por clase. Se marca como atípico lo que cae fuera de Q1 − 1,5·IQR o Q3 + 1,5·IQR.",
                  en="Boxes per class. Anything outside Q1 − 1.5·IQR or Q3 + 1.5·IQR is flagged as an outlier."),
        "read": G(es="La línea es la mediana. Una caja más alta en el grupo enfermo indica un factor de riesgo. Los atípicos se revisan, "
                     "no se borran automáticamente.",
                  en="The line is the median. A higher box in the sick group points to a risk factor. Outliers are reviewed, "
                     "not deleted automatically.")},
    "split_violins": {
        "title": G(es="Violines divididos: factor vs. variable numérica", en="Split violins: factor vs. numeric variable"),
        "what": G(es="Cada violín es la forma de la distribución; la mitad izquierda es una clase y la derecha la otra.",
                  en="Each violin is the shape of the distribution; the left half is one class and the right half the other."),
        "read": G(es="Cuando la mitad derecha (con ECV) está más arriba, esa variable es más alta en los enfermos.",
                  en="When the right half (CVD) sits higher, the variable is higher in sick people.")},
    "categorical_prevalence": {
        "title": G(es="Porcentaje de enfermos por factor", en="Share of sick people per factor"),
        "what": G(es="Prevalencia = enfermos del grupo ÷ personas del grupo, para cada nivel de cada factor.",
                  en="Prevalence = sick people in the group ÷ people in the group, for every level of every factor."),
        "read": G(es="Barras sobre la línea punteada (promedio general) marcan factores de riesgo. En este dataset fumar y la glucosa "
                     "salen al revés de lo esperado: es un sesgo conocido de los datos y por eso el modelo lleva restricciones clínicas.",
                  en="Bars above the dashed line (overall average) mark risk factors. In this dataset smoking and glucose go the "
                     "opposite way to medicine: a known bias of the data, which is why the model has clinical constraints.")},
    "scatter": {
        "title": G(es="Dispersión: presión y edad vs. IMC", en="Scatter: pressure and age vs. BMI"),
        "what": G(es="Cada punto es un paciente, coloreado por diagnóstico.", en="Each dot is a patient, coloured by diagnosis."),
        "read": G(es="Las zonas donde domina un color muestran combinaciones de valores asociadas a la enfermedad.",
                  en="Areas where one colour dominates show value combinations linked to the disease.")},
    "pairplot": {
        "title": G(es="Todas las variables de dos en dos", en="Every pair of variables"),
        "what": G(es="Matriz de dispersiones entre pares de variables; en la diagonal, la distribución de cada una.",
                  en="Grid of scatter plots between pairs of variables; on the diagonal, each variable's distribution."),
        "read": G(es="Sirve para ver de un vistazo relaciones entre variables y si alguna separa las clases.",
                  en="Shows at a glance how variables relate and whether any of them separates the classes.")},
    "correlations": {
        "title": G(es="Correlaciones de Pearson y Spearman", en="Pearson and Spearman correlations"),
        "what": G(es="Pearson mide relación lineal; Spearman, relación ordenada (usa rangos y resiste atípicos). Valores de −1 a 1.",
                  en="Pearson measures linear association; Spearman, ordered association (uses ranks, robust to outliers). From −1 to 1."),
        "read": G(es="Rojo: suben juntas. Azul: una sube y la otra baja. Blanco: sin relación. Mira la fila de la variable objetivo.",
                  en="Red: they rise together. Blue: one rises as the other falls. White: no relation. Look at the target's row.")},
    "vif_candidates": {
        "title": G(es="Multicolinealidad (VIF) de todas las variables", en="Multicollinearity (VIF) of all variables"),
        "what": G(es="VIF = 1 / (1 − R²): cuánto se puede predecir una variable a partir de las demás.",
                  en="VIF = 1 / (1 − R²): how well one variable can be predicted from the others."),
        "read": G(es="Verde < 5 bien, amarillo 5–10 moderado, rojo > 10 grave. Las variables con VIF infinito son copias "
                     "de otras y se retiran del modelo.",
                  en="Green < 5 fine, yellow 5–10 moderate, red > 10 severe. Variables with infinite VIF are copies of others "
                     "and are removed from the model.")},
    "vif_model_features": {
        "title": G(es="VIF después de quitar variables redundantes", en="VIF after removing redundant variables"),
        "what": G(es="El mismo cálculo de VIF con las variables que de verdad usa el modelo.",
                  en="The same VIF computation with the variables the model really uses."),
        "read": G(es="Todas las barras deberían quedar en verde o amarillo: ya no hay información duplicada.",
                  en="All bars should now be green or yellow: no duplicated information is left.")},
    "pca_projection": {
        "title": G(es="Mapa 2D de los datos (PCA)", en="2-D map of the data (PCA)"),
        "what": G(es="PCA resume todas las variables en dos ejes que conservan la mayor variación posible.",
                  en="PCA summarises all variables in two axes that keep as much variation as possible."),
        "read": G(es="Si los colores forman nubes separadas, las clases son fáciles de distinguir; si se mezclan, el problema es difícil.",
                  en="If the colours form separate clouds, the classes are easy to separate; if they mix, the problem is hard.")},
    # -------------------------------------------------------------- evaluation
    "metric_comparison": {
        "title": G(es="Métricas de cada modelo y del sistema combinado", en="Metrics of each model and of the combined system"),
        "what": G(es="Exactitud, precisión, sensibilidad, especificidad, F1 y AUC en datos que los modelos nunca vieron.",
                  en="Accuracy, precision, sensitivity, specificity, F1 and AUC on data the models never saw."),
        "read": G(es="Más alto es mejor. Compara la barra del sistema combinado (stacking) con las de los 4 modelos; la línea roja es la meta del 95 %.",
                  en="Higher is better. Compare the combined system (stacking) with the 4 models; the red line is the 95 % target.")},
    "confusion_matrices": {
        "title": G(es="Matrices de confusión: aciertos y errores", en="Confusion matrices: hits and errors"),
        "what": G(es="Tabla 2×2 por modelo: filas = verdad, columnas = predicción. Cada fila suma 100 %.",
                  en="2×2 table per model: rows = truth, columns = prediction. Each row adds up to 100 %."),
        "read": G(es="La diagonal son los aciertos. La celda de abajo a la izquierda son los enfermos que el modelo no detectó: "
                     "es el error más peligroso.",
                  en="The diagonal holds the hits. The bottom-left cell is the sick cases the model missed: the most dangerous error.")},
    "roc_pr_curves": {
        "title": G(es="Curvas ROC y precisión-sensibilidad", en="ROC and precision-recall curves"),
        "what": G(es="Cómo cambian aciertos y falsas alarmas al mover el umbral de decisión. El área bajo la curva (AUC) resume la calidad.",
                  en="How hits and false alarms change as the decision threshold moves. The area under the curve (AUC) sums it up."),
        "read": G(es="Una curva pegada a la esquina superior izquierda (ROC) o superior derecha (PR) es mejor. La diagonal punteada es azar (AUC 0,5).",
                  en="A curve hugging the top-left (ROC) or top-right (PR) corner is better. The dotted diagonal is chance (AUC 0.5).")},
    "calibration": {
        "title": G(es="¿Las probabilidades dicen la verdad? (calibración)", en="Do the probabilities tell the truth? (calibration)"),
        "what": G(es="Se agrupan los casos por probabilidad predicha y se compara con el porcentaje real de positivos.",
                  en="Cases are grouped by predicted probability and compared with the real share of positives."),
        "read": G(es="Sobre la diagonal, «70 %» significa que de verdad 7 de cada 10 casos así son positivos. Por debajo, el modelo exagera.",
                  en="On the diagonal, '70 %' really means 7 out of 10 such cases are positive. Below it, the model over-estimates.")},
    "meta_weights": {
        "title": G(es="Cuánto confía el sistema en cada modelo", en="How much the system trusts each model"),
        "what": G(es="Pesos β que aprendió el meta-modelo: P = σ(β₀ + Σ βₖ·pₖ). Son ≥ 0 por diseño.",
                  en="Weights β learned by the meta-model: P = σ(β₀ + Σ βₖ·pₖ). They are ≥ 0 by design."),
        "read": G(es="Barra más alta = el resultado final escucha más a ese modelo. Un peso 0 significa que su información ya la aportan los otros.",
                  en="Taller bar = the final answer listens more to that model. A weight of 0 means the others already carry its information.")},
    "model_diversity": {
        "title": G(es="¿Los 4 modelos piensan distinto?", en="Do the 4 models think differently?"),
        "what": G(es="Izquierda: correlación entre las probabilidades de cada par de modelos. Derecha: % de casos en que discrepan.",
                  en="Left: correlation between the probabilities of each pair of models. Right: % of cases where they disagree."),
        "read": G(es="Combinar modelos solo ayuda si a veces discrepan: correlaciones menores a 1 y desacuerdos mayores a 0 indican modelos complementarios.",
                  en="Combining models only helps if they sometimes disagree: correlations below 1 and non-zero disagreement mean complementary models.")},
    "probability_distribution": {
        "title": G(es="Probabilidad final según la clase real", en="Final probability by true class"),
        "what": G(es="Histograma de la probabilidad que da el sistema, un color por clase real. La franja gris es la zona de duda.",
                  en="Histogram of the system's probability, one colour per true class. The grey band is the doubt zone."),
        "read": G(es="Lo ideal: un color pegado a 0 y el otro a 1. Donde los colores se cruzan están los errores; los casos en la franja gris se remiten al médico.",
                  en="Ideal: one colour near 0 and the other near 1. Errors live where colours overlap; cases in the grey band go to a clinician.")},
    "coverage_curve": {
        "title": G(es="Exactitud si el sistema solo responde lo que tiene claro", en="Accuracy if the system only answers clear cases"),
        "what": G(es="Se ordenan los casos de más a menos seguros; para cada porcentaje respondido (cobertura) se calcula la exactitud.",
                  en="Cases are sorted from most to least confident; for each share answered (coverage) the accuracy is computed."),
        "read": G(es="Hacia la izquierda el sistema responde menos casos y acierta más. Donde la curva cruza la línea del 95 % se lee qué parte "
                     "de los casos puede resolverse sola con esa exactitud.",
                  en="Towards the left the system answers fewer cases and gets more right. Where the curve crosses the 95 % line tells "
                     "which share of cases it can solve on its own at that accuracy.")},
    "permutation_importance": {
        "title": G(es="¿En qué datos se apoya el sistema?", en="Which inputs does the system rely on?"),
        "what": G(es="Importancia por permutación: cuánto cae la exactitud al desordenar al azar una variable.",
                  en="Permutation importance: how much accuracy drops when one input is randomly shuffled."),
        "read": G(es="Barra más larga = el sistema depende más de ese dato. Barras cerca de 0 = se podría quitar sin perder exactitud.",
                  en="Longer bar = the system depends more on that input. Bars near 0 = it could be removed without losing accuracy.")},
    "logistic_coefficients": {
        "title": G(es="Dirección de cada factor de riesgo", en="Direction of each risk factor"),
        "what": G(es="Coeficientes de la regresión logística por cada desviación estándar de la variable (restringidos al signo clínico).",
                  en="Logistic-regression coefficients per standard deviation of each input (constrained to the clinical sign)."),
        "read": G(es="Naranja = sube el riesgo, azul = lo baja. Cuanto más larga la barra, mayor el efecto.",
                  en="Orange = raises the risk, blue = lowers it. The longer the bar, the stronger the effect.")},
    "plausibility_audit": {
        "title": G(es="Auditoría clínica: ¿el riesgo se mueve en la dirección correcta?", en="Clinical audit: does risk move the right way?"),
        "what": G(es="Para pacientes reales se sube un factor (presión, peso, colesterol…) dejando lo demás igual y se cuenta a cuántos "
                     "les baja el riesgo más de 1 punto.",
                  en="For real patients one factor (pressure, weight, cholesterol…) is raised with everything else fixed, counting "
                     "for how many the risk falls by more than 1 point."),
        "read": G(es="0 % (blanco) = siempre coherente con la medicina. Rojo = el modelo contradice la clínica. La columna «Stacking» es la respuesta final.",
                  en="0 % (white) = always consistent with medicine. Red = the model contradicts clinical knowledge. The 'Stacking' column is the final answer.")},
    "kmeans_k_scan": {
        "title": G(es="¿Cuántos grupos de pacientes? (K-means)", en="How many patient groups? (K-means)"),
        "what": G(es="Método del codo (inercia J) y coeficiente de silueta para k = 2…8 grupos.",
                  en="Elbow method (inertia J) and silhouette coefficient for k = 2…8 groups."),
        "read": G(es="Se busca el «codo» donde J deja de bajar rápido y una silueta alta. Se eligió k = 3 (riesgo bajo, medio, alto).",
                  en="Look for the 'elbow' where J stops falling fast and a high silhouette. k = 3 was chosen (low, medium, high risk).")},
    "risk_segments_pca": {
        "title": G(es="Grupos de riesgo en el mapa PCA", en="Risk groups on the PCA map"),
        "what": G(es="Pacientes de prueba coloreados por el grupo que les asignó K-means (sin usar el diagnóstico).",
                  en="Test patients coloured by the group K-means gave them (without using the diagnosis)."),
        "read": G(es="Cada color es un perfil parecido de pacientes; los grupos ocupan zonas distintas del mapa.",
                  en="Each colour is a similar patient profile; the groups occupy different areas of the map.")},
    "risk_segments_profile": {
        "title": G(es="Perfil de cada grupo y su porcentaje de enfermos", en="Profile of each group and its share of sick people"),
        "what": G(es="Izquierda: promedio de cada variable por grupo. Derecha: porcentaje real con ECV en cada grupo.",
                  en="Left: mean of each variable per group. Right: real share with CVD in each group."),
        "read": G(es="El grupo de riesgo alto debe ser mayor, con presión más alta y más enfermos. Confirma que los grupos tienen sentido clínico.",
                  en="The high-risk group should be older, with higher pressure and more disease. It confirms the groups make clinical sense.")},
    "segments_vs_ensemble": {
        "title": G(es="Los grupos y el modelo coinciden", en="The groups and the model agree"),
        "what": G(es="Probabilidad que da el sistema supervisado dentro de cada grupo no supervisado.",
                  en="Probability given by the supervised system inside each unsupervised group."),
        "read": G(es="Las cajas deben subir de bajo a alto: dos métodos independientes llegan a la misma conclusión.",
                  en="Boxes should rise from low to high: two independent methods reach the same conclusion.")},
    "simulation_archetypes": {
        "title": G(es="Simulación: pacientes hipotéticos a través del sistema", en="Simulation: hypothetical patients through the system"),
        "what": G(es="Cada fila es un paciente inventado; cada columna, la probabilidad que da cada modelo y, al final, la respuesta única.",
                  en="Each row is an invented patient; each column the probability from each model and, last, the single answer."),
        "read": G(es="Azul = bajo riesgo, rojo = alto. Permite seguir caso por caso cómo los 4 modelos se combinan en 1 respuesta. "
                     "«[refer]» = se remite al médico.",
                  en="Blue = low risk, red = high. Follow case by case how the 4 models combine into 1 answer. '[refer]' = sent to a clinician.")},
    "simulation_test_patients": {
        "title": G(es="Simulación con pacientes reales no vistos", en="Simulation with unseen real patients"),
        "what": G(es="Igual que la anterior pero con pacientes reales del conjunto de prueba, con su diagnóstico verdadero.",
                  en="Same as above with real test-set patients and their true diagnosis."),
        "read": G(es="«(ok)» = el sistema acertó, «(WRONG)» = se equivocó. Es la trazabilidad completa de cada predicción.",
                  en="'(ok)' = the system was right, '(WRONG)' = it was wrong. It is the full trace of each prediction.")},
    # ---------------------------------------------------------------------- ECG
    "ecg_etl_example": {
        "title": G(es="Del ECG crudo a datos limpios (ETL)", en="From raw ECG to clean data (ETL)"),
        "what": G(es="Arriba la señal original con ruido y línea base que se mueve; abajo la señal filtrada (0,5–40 Hz) y "
                     "remuestreada a 128 Hz, con la etiqueta del cardiólogo en cada latido.",
                  en="Top: the original signal with noise and wandering baseline; bottom: the filtered (0.5–40 Hz) and "
                     "resampled 128 Hz signal, with the cardiologist's label on each beat."),
        "read": G(es="Las letras rojas son latidos anormales (V ventricular, A/S supraventricular, F fusión). Así una señal no estructurada se vuelve filas de una tabla.",
                  en="Red letters are abnormal beats (V ventricular, A/S supraventricular, F fusion). This is how an unstructured signal becomes table rows.")},
    "beat_counts": {
        "title": G(es="Latidos por tipo y base de datos", en="Beats per type and database"),
        "what": G(es="Cantidad de latidos de cada clase AAMI (N normal, S supraventricular, V ventricular, F fusión), en escala logarítmica.",
                  en="Number of beats of each AAMI class (N normal, S supraventricular, V ventricular, F fusion), on a log scale."),
        "read": G(es="Cada marca del eje es ×10. Los normales son la gran mayoría: por eso hay que balancear.",
                  en="Each axis tick is ×10. Normal beats are the vast majority: that is why balancing is needed.")},
    "mean_beats": {
        "title": G(es="Forma promedio de cada tipo de latido", en="Average shape of each beat type"),
        "what": G(es="Promedio ± 1 desviación estándar de la forma del latido; el tiempo 0 es el pico R.",
                  en="Mean ± 1 standard deviation of the beat shape; time 0 is the R peak."),
        "read": G(es="El ventricular (V) es ancho y deforme. El supraventricular (S) se parece al normal: por eso también se usa el ritmo (cuándo llega).",
                  en="The ventricular beat (V) is wide and deformed. The supraventricular (S) looks normal: that is why rhythm (when it arrives) is also used.")},
    "rr_features": {
        "title": G(es="Ritmo y forma: normal vs. anormal", en="Rhythm and shape: normal vs. abnormal"),
        "what": G(es="Violines de las variables de ritmo (intervalos RR), amplitud y comparación con el latido habitual del paciente.",
                  en="Violins of the rhythm (RR interval), amplitude and patient-relative features."),
        "read": G(es="Ejemplo: los anormales tienen rr_ratio < 1 (llegan antes de tiempo) y menor parecido con el latido habitual (template_corr).",
                  en="Example: abnormal beats have rr_ratio < 1 (they arrive early) and lower similarity to the usual beat (template_corr).")},
    "vif_rhythm_features": {
        "title": G(es="Multicolinealidad de las variables de ritmo", en="Multicollinearity of the rhythm features"),
        "what": G(es="VIF = 1 / (1 − R²) de las variables de ritmo y amplitud.", en="VIF = 1 / (1 − R²) of the rhythm and amplitude features."),
        "read": G(es="Verde < 5 bien, rojo > 10 grave. Los modelos de árboles y la regularización toleran la redundancia moderada.",
                  en="Green < 5 fine, red > 10 severe. Tree models and regularisation tolerate moderate redundancy.")},
    "simulation_test_beats": {
        "title": G(es="Simulación: latidos del conjunto de prueba", en="Simulation: held-out test beats"),
        "what": G(es="Latidos de cada tipo (N, S, V, F) que no se usaron para entrenar (división por latido), con la probabilidad "
                     "de cada modelo y la respuesta final.",
                  en="Beats of each type (N, S, V, F) not used for training (split by beat), with each model's probability and "
                     "the final answer."),
        "read": G(es="Azul = normal, rojo = anormal. «(WRONG)» marca los errores; los S son los más difíciles.",
                  en="Blue = normal, red = abnormal. '(WRONG)' marks the errors; S beats are the hardest.")},
    "leakage_check": {
        "title": G(es="Prueba de fuga: ¿el modelo memoriza pacientes?", en="Leakage check: does the model memorise patients?"),
        "what": G(es="Exactitud de cada modelo con dos formas de separar los datos: al azar por latido (el mismo paciente puede quedar "
                     "en entrenamiento y prueba) y por paciente (los de prueba nunca se vieron).",
                  en="Accuracy of each model with two ways of splitting the data: at random by beat (the same patient can be in "
                     "training and test) and by patient (test patients were never seen)."),
        "read": G(es="La cifra oficial es la barra gris (división por latido, el protocolo estándar de MIT-BIH). La barra de color es "
                     "un análisis adicional: muestra cuánto baja la exactitud cuando el paciente es totalmente nuevo.",
                  en="The official figure is the grey bar (split by beat, the standard MIT-BIH protocol). The coloured bar is an "
                     "additional analysis: it shows how much accuracy drops when the patient is completely new.")},
    "patient_cv": {
        "title": G(es="Validación cruzada por paciente (5 particiones)", en="Patient-wise cross-validation (5 folds)"),
        "what": G(es="Se divide a los pacientes en 5 grupos; cada vez se entrena con 4 y se prueba con el que queda, sin repetir pacientes.",
                  en="Patients are split into 5 groups; each time the system trains on 4 and is tested on the remaining one, never sharing patients."),
        "read": G(es="Cada punto es una partición. Si todos quedan cerca, el resultado no depende de qué pacientes tocaron en la prueba.",
                  en="Each dot is one fold. If they all sit close together, the result does not depend on which patients ended up in the test.")},
    "inter_patient_metrics": {
        "title": G(es="Pacientes nuevos con la proporción real de latidos", en="New patients with the real share of beats"),
        "what": G(es="Métricas en pacientes no vistos sin balancear (≈ 11 % de latidos anormales, como en un Holter real), con la corrección "
                     "de prevalencia de Bayes.",
                  en="Metrics on unseen patients without balancing (≈ 11 % abnormal beats, as in a real Holter), with Bayes' prevalence correction."),
        "read": G(es="Aquí la exactitud sola engaña (decir siempre «normal» daría 89 %): mira la exactitud balanceada y la sensibilidad.",
                  en="Here accuracy alone misleads (always saying 'normal' would score 89 %): look at balanced accuracy and sensitivity.")},
    "automatic_detection": {
        "title": G(es="Prueba de despliegue: latidos detectados automáticamente", en="Deployment check: automatically detected beats"),
        "what": G(es="Compara clasificar los latidos marcados por el cardiólogo con los que encuentra el detector automático (XQRS), como hace la app.",
                  en="Compares classifying the beats marked by the cardiologist with the ones the automatic detector (XQRS) finds, as the app does."),
        "read": G(es="Si las barras son casi iguales, la app funciona con ECG sin anotar sin perder exactitud.",
                  en="If both bars are almost equal, the app works on unannotated ECGs without losing accuracy.")},
    "monitoring_session": {
        "title": G(es="Simulación de monitoreo remoto", en="Remote-monitoring simulation"),
        "what": G(es="Un tramo del ECG de un paciente no visto, con lo que predijo el sistema en cada latido.",
                  en="A stretch of an unseen patient's ECG with the system's prediction for every beat."),
        "read": G(es="Azul = predicho normal, rojo = predicho anormal, X = error. Debajo de cada latido: su clase real y la probabilidad.",
                  en="Blue = predicted normal, red = predicted abnormal, X = error. Under each beat: its true class and the probability.")},
    "ecg:metric_comparison": {
        "title": G(es="Métricas de cada modelo en el conjunto de prueba", en="Metrics of each model on the test set"),
        "what": G(es="Exactitud, precisión, sensibilidad, especificidad, F1 y AUC en latidos de prueba que no se usaron para "
                     "entrenar, pero de pacientes que sí aportaron otros latidos (división por latido).",
                  en="Accuracy, precision, sensitivity, specificity, F1 and AUC on test beats not used for training, but from "
                     "patients who did give other beats to training (split by beat)."),
        "read": G(es="Más alto es mejor; la línea roja es la meta del 95 %. Es la evaluación oficial (protocolo estándar de MIT-BIH). "
                     "El análisis adicional con pacientes totalmente nuevos está en la figura «Prueba de fuga».",
                  en="Higher is better; the red line is the 95 % target. This is the official evaluation (standard MIT-BIH "
                     "protocol). The additional analysis on completely new patients is in the 'Leakage check' figure.")},
    "ecg:distributions": {
        "title": G(es="Distribución de las variables de ritmo por clase", en="Distribution of rhythm features by class"),
        "what": G(es="Histograma + curva KDE de las variables de ritmo y de parecido con el latido habitual.",
                  en="Histogram + KDE curve of the rhythm and similarity-to-usual-beat features."),
        "read": G(es="Las curvas separadas indican variables que distinguen latidos normales de anormales.",
                  en="Curves that sit apart mark features that tell normal from abnormal beats.")},
}


def guide_for(slug: str, tier: str = "") -> dict | None:
    """Guide entry for a figure slug (tier-specific text first)."""
    return GUIDE.get(f"{tier}:{slug}") or GUIDE.get(slug)
