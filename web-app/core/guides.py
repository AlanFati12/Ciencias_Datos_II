"""«¿Cómo leer esta gráfica?» — plain-language guide for every chart of the app.

Each entry answers three questions a non-expert asks in front of a chart:
    what  - what am I looking at?
    read  - how do I read it (what to look at first)?
    note  - one practical example or warning (optional).
Shown in a popover next to the chart title (ui.chart_header). The figures of the
ML pipeline carry their own guide in run_manifest.json (cardio_ml/figure_guide.py).
"""
from __future__ import annotations

from .i18n import lang

GUIDES: dict[str, dict[str, dict[str, str]]] = {
    "result": {
        "what": {"es": "El porcentaje es la probabilidad que calcula el sistema. El color resume qué hacer: verde bajo, "
                       "amarillo «zona de duda» (35–65 %) y rojo alto.",
                 "en": "The percentage is the probability computed by the system. The colour sums up what to do: green "
                       "low, yellow 'doubt zone' (35–65 %) and red high."},
        "read": {"es": "La barra de colores ubica al paciente entre 0 % y 100 %. Si el resultado cae en la zona de duda, o si los "
                       "4 modelos se dividen 2 contra 2, el sistema no decide solo y recomienda que lo revise un profesional.",
                 "en": "The coloured bar places the patient between 0 % and 100 %. If the result falls in the doubt zone, or "
                       "the 4 models split 2 against 2, the system does not decide alone and recommends a professional review."},
        "note": {"es": "Es una herramienta de apoyo: no es un diagnóstico.", "en": "It is a support tool, not a diagnosis."}},
    "factors": {
        "what": {"es": "Cuánto sube (↑ rojo) o baja (↓ verde) el riesgo cada dato del paciente, en puntos porcentuales.",
                 "en": "How much each piece of patient data raises (↑ red) or lowers (↓ green) the risk, in percentage points."},
        "read": {"es": "Se calcula cambiando un dato por el valor típico de la población y viendo cuánto cambia el resultado. "
                       "Los primeros de la lista son los que más pesan para este paciente.",
                 "en": "It is computed by replacing one value with the population's typical value and measuring how much the "
                       "result changes. The first items are the ones that matter most for this patient."},
        "note": {"es": "Ejemplo: «Presión arterial +12 puntos» = con presión normal su riesgo sería 12 puntos menor.",
                 "en": "Example: 'Blood pressure +12 points' = with a normal pressure the risk would be 12 points lower."}},
    "opinions": {
        "what": {"es": "La probabilidad que da cada uno de los 4 modelos por separado y, en negro, la respuesta final que los combina.",
                 "en": "The probability given by each of the 4 models on its own and, in black, the final answer that combines them."},
        "read": {"es": "Barras a la derecha de la línea del 50 % votan «sí»; a la izquierda, «no». La respuesta final no es un "
                       "promedio simple: el meta-modelo da más peso a los modelos que mejor funcionaron en el entrenamiento.",
                 "en": "Bars right of the 50 % line vote 'yes'; left of it, 'no'. The final answer is not a plain average: the "
                       "meta-model gives more weight to the models that worked best in training."},
        "note": {"es": "Si los modelos no se ponen de acuerdo (2 contra 2), el caso se remite al médico.",
                 "en": "If the models disagree (2 against 2), the case is referred to a clinician."}},
    "sweep": {
        "what": {"es": "Cómo cambiaría el riesgo de este paciente si solo cambiara su presión sistólica y todo lo demás quedara igual.",
                 "en": "How this patient's risk would change if only the systolic pressure changed and everything else stayed the same."},
        "read": {"es": "El punto rojo es el valor actual. Si la línea sube hacia la derecha, bajar la presión reduciría el riesgo. "
                       "La franja amarilla es la zona de duda.",
                 "en": "The red dot is the current value. If the line rises to the right, lowering the pressure would reduce "
                       "the risk. The yellow band is the doubt zone."},
        "note": {"es": "La curva nunca baja cuando la presión sube: el modelo tiene restricciones clínicas que lo impiden.",
                 "en": "The curve never falls as pressure rises: the model has clinical constraints that prevent it."}},
    "ecg_strip": {
        "what": {"es": "Un tramo del electrocardiograma ya filtrado. Encima de cada latido hay una marca con lo que decidió el sistema.",
                 "en": "A stretch of the filtered electrocardiogram. Above each beat a marker shows what the system decided."},
        "read": {"es": "● azul = latido normal, ◆ rojo = latido anormal, ○ gris = no se pudo clasificar (ruido o pausa). Pasa el "
                       "mouse sobre una marca para ver su probabilidad. Usa «Siguiente anormal» para saltar entre los rojos.",
                 "en": "● blue = normal beat, ◆ red = abnormal beat, ○ grey = could not be classified (noise or pause). Hover "
                       "a marker to see its probability. Use 'Next abnormal' to jump between the red ones."},
        "note": {"es": "Puedes arrastrar sobre la gráfica para hacer zoom; doble clic para volver.",
                 "en": "Drag over the chart to zoom in; double-click to reset."}},
    "beat_shape": {
        "what": {"es": "La forma del latido elegido (línea roja) sobre la forma habitual de los latidos de este paciente (punteada).",
                 "en": "The shape of the chosen beat (red line) over the shape of this patient's usual beat (dotted)."},
        "read": {"es": "Si las dos líneas casi coinciden, el latido tiene forma normal; si la roja es más ancha o deforme, es "
                       "típico de un latido ventricular. Un latido supraventricular tiene forma casi normal pero llega antes de tiempo.",
                 "en": "If both lines almost match, the beat has a normal shape; if the red one is wider or deformed, it is "
                       "typical of a ventricular beat. A supraventricular beat looks almost normal but arrives early."}},
    "tachogram": {
        "what": {"es": "El ritmo de todo el registro: cada punto es un latido; la altura es el tiempo desde el latido anterior.",
                 "en": "The rhythm of the whole recording: each dot is a beat; its height is the time since the previous beat."},
        "read": {"es": "Una banda plana = ritmo regular. Puntos que caen por debajo de la banda son latidos que llegaron antes "
                       "de tiempo (prematuros); los que suben son pausas. Más alto = corazón más lento.",
                 "en": "A flat band = regular rhythm. Dots that drop below the band are beats that came early (premature); "
                       "dots above it are pauses. Higher = slower heart."}},
    "accuracy": {
        "what": {"es": "Qué porcentaje de casos nunca vistos acierta cada modelo y el sistema combinado.",
                 "en": "Which share of never-seen cases each model and the combined system get right."},
        "read": {"es": "Barra más larga = más aciertos. La línea roja es la meta del 95 % de la rúbrica.",
                 "en": "Longer bar = more hits. The red line is the 95 % rubric target."}},
    "folds": {
        "what": {"es": "Exactitud en cada una de las 5 particiones por paciente: cada vez se prueba con pacientes que el modelo nunca vio.",
                 "en": "Accuracy in each of the 5 patient-wise folds: each time the test uses patients the model never saw."},
        "read": {"es": "Si los 5 puntos están juntos, el resultado no depende de la suerte de qué pacientes tocaron en la prueba. "
                       "La línea negra es el promedio.",
                 "en": "If the 5 dots sit together, the result does not depend on which patients happened to be in the test. "
                       "The black line is the mean."}},
    "figure": {
        "what": {"es": "Cada gráfica del pipeline trae su guía: qué muestra y cómo leerla.",
                 "en": "Each pipeline figure comes with its guide: what it shows and how to read it."},
        "read": {"es": "Elige una gráfica en la lista; la explicación aparece al lado.",
                 "en": "Pick a figure from the list; its explanation appears next to it."}},
}


def guide(key: str) -> dict[str, str]:
    entry = GUIDES[key]
    return {part: text[lang()] for part, text in entry.items()}


# ------------------------------------------------------------------ glossary
GLOSSARY: list[tuple[dict[str, str], dict[str, str]]] = [
    ({"es": "Probabilidad", "en": "Probability"},
     {"es": "Número de 0 % a 100 % que dice qué tan probable es algo. «45 %» = de cada 100 casos parecidos, unos 45 lo tienen.",
      "en": "A number from 0 % to 100 % that says how likely something is. '45 %' = out of 100 similar cases, about 45 have it."}),
    ({"es": "Exactitud (accuracy)", "en": "Accuracy"},
     {"es": "Porcentaje de respuestas correctas sobre casos que el modelo no usó para aprender.",
      "en": "Share of correct answers on cases the model did not use to learn."}),
    ({"es": "Sensibilidad", "en": "Sensitivity"},
     {"es": "De los enfermos (o latidos anormales) reales, cuántos detecta el sistema. Alta sensibilidad = pocos casos perdidos.",
      "en": "Of the truly sick people (or abnormal beats), how many the system finds. High sensitivity = few missed cases."}),
    ({"es": "Especificidad", "en": "Specificity"},
     {"es": "De los sanos (o latidos normales) reales, cuántos descarta correctamente. Alta especificidad = pocas falsas alarmas.",
      "en": "Of the truly healthy people (or normal beats), how many it correctly clears. High specificity = few false alarms."}),
    ({"es": "ROC-AUC", "en": "ROC-AUC"},
     {"es": "Qué tan bien ordena el modelo a enfermos por encima de sanos (0,5 = azar, 1 = perfecto).",
      "en": "How well the model ranks sick above healthy cases (0.5 = chance, 1 = perfect)."}),
    ({"es": "Dataset balanceado", "en": "Balanced dataset"},
     {"es": "Mismo número de casos de cada clase (50/50), para que la exactitud no se infle respondiendo siempre lo más común.",
      "en": "Same number of cases of each class (50/50), so accuracy cannot be inflated by always answering the common class."}),
    ({"es": "Stacking (ensamble apilado)", "en": "Stacking"},
     {"es": "Forma de combinar modelos: un meta-modelo aprende cuánto confiar en cada uno y entrega una sola respuesta.",
      "en": "A way to combine models: a meta-model learns how much to trust each one and gives a single answer."}),
    ({"es": "Zona de duda / remisión", "en": "Doubt zone / referral"},
     {"es": "Si la probabilidad final está entre 35 % y 65 %, o los 4 modelos empatan 2 a 2, el sistema no decide y recomienda revisión médica.",
      "en": "If the final probability is between 35 % and 65 %, or the 4 models split 2–2, the system does not decide and recommends a medical review."}),
    ({"es": "Electrocardiograma (ECG)", "en": "Electrocardiogram (ECG)"},
     {"es": "Registro de la actividad eléctrica del corazón. Un Holter lo graba durante horas.",
      "en": "Recording of the heart's electrical activity. A Holter records it for hours."}),
    ({"es": "Intervalo RR", "en": "RR interval"},
     {"es": "Tiempo entre dos latidos (entre dos picos R). Un latido que llega antes de tiempo tiene un RR corto.",
      "en": "Time between two beats (between two R peaks). A beat that comes early has a short RR."}),
    ({"es": "Clases AAMI (N, S, V, F)", "en": "AAMI classes (N, S, V, F)"},
     {"es": "N normal · S supraventricular (nace arriba de los ventrículos, llega antes) · V ventricular (nace en los ventrículos, "
            "forma ancha) · F fusión (mezcla de normal y ventricular).",
      "en": "N normal · S supraventricular (starts above the ventricles, comes early) · V ventricular (starts in the ventricles, "
            "wide shape) · F fusion (mix of normal and ventricular)."}),
    ({"es": "Protocolo intra-paciente vs. inter-paciente", "en": "Intra-patient vs. inter-patient protocol"},
     {"es": "Intra-paciente (estándar en MIT-BIH, la cifra oficial): los latidos de prueba no se usaron para entrenar, pero sus "
            "pacientes sí aportaron otros latidos. Inter-paciente (análisis adicional): los pacientes de prueba nunca se vieron; "
            "es más estricto y da una cifra menor.",
      "en": "Intra-patient (standard for MIT-BIH, the official figure): the test beats were not used for training, but their "
            "patients gave other beats. Inter-patient (additional analysis): the test patients were never seen; it is stricter "
            "and gives a lower figure."}),
    ({"es": "Fuga de datos (data leakage)", "en": "Data leakage"},
     {"es": "Cuando información de la prueba se cuela en el entrenamiento y hace ver el modelo mejor de lo que es.",
      "en": "When information from the test leaks into training and makes the model look better than it is."}),
    ({"es": "Trazabilidad", "en": "Traceability"},
     {"es": "Registro de cada paso: qué datos entraron (con su huella SHA-256), qué filas se quitaron y por qué, parámetros, "
            "métricas, gráficas y modelos exportados.",
      "en": "Record of every step: which data went in (with its SHA-256 fingerprint), which rows were removed and why, "
            "parameters, metrics, figures and exported models."}),
]


def glossary() -> list[tuple[str, str]]:
    return [(term[lang()], text[lang()]) for term, text in GLOSSARY]
