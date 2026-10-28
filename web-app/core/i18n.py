"""Bilingual interface (Spanish / English). Every visible text goes through t().

Writing rules: plain language first, technical words only where they are
explained. Texts may contain simple HTML (<b>) because they are trusted.
"""
from __future__ import annotations

import streamlit as st

LANGS = {"es": "Español", "en": "English"}
DEFAULT_LANG = "es"

T: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------ app shell
    "app.tagline": {"es": "Apoyo a la decisión cardiovascular", "en": "Cardiovascular decision support"},
    "app.language": {"es": "Idioma", "en": "Language"},
    "app.side_note": {
        "es": "Prototipo académico · Ciencia de Datos II, Universidad Surcolombiana. No reemplaza el criterio médico.",
        "en": "Academic prototype · Data Science II, Universidad Surcolombiana. It does not replace medical judgement."},
    "nav.home": {"es": "Inicio", "en": "Home"},
    "nav.risk": {"es": "Evaluar paciente", "en": "Assess a patient"},
    "nav.ecg": {"es": "Analizar ECG", "en": "Analyse an ECG"},
    "nav.history": {"es": "Historial", "en": "History"},
    "nav.batch": {"es": "Varios pacientes", "en": "Several patients"},
    "nav.how": {"es": "Cómo funciona", "en": "How it works"},
    "nav.sec_patients": {"es": "Pacientes", "en": "Patients"},
    "nav.sec_records": {"es": "Registros", "en": "Records"},
    "nav.sec_info": {"es": "Información", "en": "Information"},

    # --------------------------------------------------------------- common
    "common.models_missing": {
        "es": "No se encontraron los modelos del nivel «{tier}». Entrénalos primero con el pipeline de ML:",
        "en": "The models of the '{tier}' tier were not found. Train them first with the ML pipeline:"},
    "common.yes": {"es": "Sí", "en": "Yes"},
    "common.no": {"es": "No", "en": "No"},
    "common.patient_code": {"es": "Código del paciente", "en": "Patient code"},
    "common.patient_code_ph": {"es": "Opcional, p. ej. P-001", "en": "Optional, e.g. P-001"},
    "common.patient_code_help": {"es": "Usa un código, no el nombre, para proteger la privacidad del paciente.",
                                 "en": "Use a code, not the name, to protect the patient's privacy."},
    "common.save": {"es": "Guardar en el historial", "en": "Save to history"},
    "common.saved": {"es": "Guardado en el historial (registro #{id}).", "en": "Saved to history (record #{id})."},
    "common.pdf": {"es": "Descargar reporte PDF", "en": "Download PDF report"},
    "common.csv": {"es": "Descargar CSV", "en": "Download CSV"},
    "common.disclaimer": {
        "es": "Prototipo académico de apoyo a la decisión. No es un dispositivo médico certificado y no reemplaza "
              "el criterio de un profesional de la salud.",
        "en": "Academic decision-support prototype. It is not a certified medical device and does not replace the "
              "judgement of a health professional."},
    "common.edit": {"es": "Editar datos", "en": "Edit data"},
    "common.new": {"es": "Nueva evaluación", "en": "New assessment"},
    "common.points": {"es": "{v:+.0f} puntos", "en": "{v:+.0f} points"},

    # ----------------------------------------------------------------- inputs
    "in.age": {"es": "Edad (años)", "en": "Age (years)"},
    "in.sex": {"es": "Sexo", "en": "Sex"},
    "in.female": {"es": "Mujer", "en": "Female"},
    "in.male": {"es": "Hombre", "en": "Male"},
    "in.height": {"es": "Estatura (cm)", "en": "Height (cm)"},
    "in.weight": {"es": "Peso (kg)", "en": "Weight (kg)"},
    "in.ap_hi": {"es": "Sistólica (alta)", "en": "Systolic (upper)"},
    "in.ap_lo": {"es": "Diastólica (baja)", "en": "Diastolic (lower)"},
    "in.cholesterol": {"es": "Colesterol", "en": "Cholesterol"},
    "in.glucose": {"es": "Glucosa", "en": "Glucose"},
    "in.smoke": {"es": "Fuma", "en": "Smokes"},
    "in.alcohol": {"es": "Consume alcohol", "en": "Drinks alcohol"},
    "in.active": {"es": "Hace actividad física", "en": "Is physically active"},
    "level.1": {"es": "Normal", "en": "Normal"},
    "level.2": {"es": "Alto", "en": "High"},
    "level.3": {"es": "Muy alto", "en": "Very high"},
    "bp_stage.0": {"es": "Normal", "en": "Normal"},
    "bp_stage.1": {"es": "Elevada", "en": "Elevated"},
    "bp_stage.2": {"es": "Hipertensión grado 1", "en": "Hypertension stage 1"},
    "bp_stage.3": {"es": "Hipertensión grado 2", "en": "Hypertension stage 2"},
    "bp_stage.4": {"es": "Crisis hipertensiva", "en": "Hypertensive crisis"},
    "bmi_cat.0": {"es": "Bajo peso", "en": "Underweight"},
    "bmi_cat.1": {"es": "Peso normal", "en": "Normal weight"},
    "bmi_cat.2": {"es": "Sobrepeso", "en": "Overweight"},
    "bmi_cat.3": {"es": "Obesidad", "en": "Obesity"},
    "segment.Low": {"es": "bajo", "en": "low"},
    "segment.Medium": {"es": "medio", "en": "medium"},
    "segment.High": {"es": "alto", "en": "high"},
    "model.LR": {"es": "Regresión logística", "en": "Logistic regression"},
    "model.KNN": {"es": "K vecinos más cercanos", "en": "K-nearest neighbours"},
    "model.RF": {"es": "Bosque aleatorio", "en": "Random forest"},
    "model.SVM": {"es": "Máquina de vectores de soporte", "en": "Support vector machine"},
    "model.final": {"es": "Respuesta final", "en": "Final answer"},

    # ------------------------------------------------------------------- home
    "home.title": {"es": "Evaluación cardiovascular asistida por inteligencia artificial",
                   "en": "AI-assisted cardiovascular assessment"},
    "home.subtitle": {
        "es": "Dos herramientas para profesionales de la salud en telemedicina: estimar el riesgo de un paciente "
              "con un examen básico y revisar su electrocardiograma latido por latido.",
        "en": "Two tools for health professionals working in telemedicine: estimate a patient's risk from a basic "
              "exam, and review their electrocardiogram beat by beat."},
    "home.risk_title": {"es": "Evaluar a un paciente", "en": "Assess a patient"},
    "home.risk_text": {
        "es": "Con los datos de un examen básico (edad, presión, colesterol…) obtén la probabilidad de enfermedad "
              "cardiovascular y qué la explica.",
        "en": "With the data of a basic exam (age, blood pressure, cholesterol…) get the probability of "
              "cardiovascular disease and what explains it."},
    "home.risk_meta": {"es": "≈ 1 minuto · 11 datos", "en": "≈ 1 minute · 11 fields"},
    "home.risk_btn": {"es": "Empezar evaluación", "en": "Start assessment"},
    "home.ecg_title": {"es": "Analizar un electrocardiograma", "en": "Analyse an electrocardiogram"},
    "home.ecg_text": {
        "es": "Sube el ECG de un Holter o de un dispositivo portátil y descubre qué latidos son anormales.",
        "en": "Upload the ECG of a Holter or a wearable device and find out which beats are abnormal."},
    "home.ecg_meta": {"es": "Incluye ejemplos · CSV o WFDB", "en": "Examples included · CSV or WFDB"},
    "home.ecg_btn": {"es": "Analizar un ECG", "en": "Analyse an ECG"},
    "home.how_title": {"es": "¿Cómo funciona?", "en": "How does it work?"},
    "home.how1_t": {"es": "Ingresas los datos", "en": "You enter the data"},
    "home.how1": {"es": "Del examen del paciente o de su electrocardiograma.", "en": "From the patient's exam or their ECG."},
    "home.how2_t": {"es": "Cuatro modelos de IA opinan", "en": "Four AI models give an opinion"},
    "home.how2": {"es": "Cada uno analiza el caso por separado y da su propia probabilidad.",
                  "en": "Each one analyses the case on its own and gives its own probability."},
    "home.how3_t": {"es": "Un modelo final decide", "en": "A final model decides"},
    "home.how3": {"es": "Combina las cuatro opiniones en una sola respuesta y te avisa cuando hay duda.",
                  "en": "It combines the four opinions into one answer and warns you when it is unsure."},
    "home.trust": {
        "es": "Entrenado con <b>{patients:,}</b> pacientes y <b>{beats:,}</b> latidos anotados por expertos.",
        "en": "Trained on <b>{patients:,}</b> patients and <b>{beats:,}</b> expert-annotated heartbeats."},
    "home.trust_link": {"es": "Ver resultados y cómo funciona", "en": "See results and how it works"},
    "home.more": {"es": "También puedes", "en": "You can also"},
    "home.more_history": {"es": "Ver el historial de evaluaciones", "en": "See the assessment history"},
    "home.more_batch": {"es": "Evaluar varios pacientes con un archivo", "en": "Assess several patients from a file"},

    # ------------------------------------------------------------------- risk
    "risk.title": {"es": "Evaluar a un paciente", "en": "Assess a patient"},
    "risk.subtitle": {"es": "Completa los datos de un examen médico básico. Toma menos de un minuto.",
                      "en": "Fill in the data of a basic medical exam. It takes less than a minute."},
    "risk.step1": {"es": "Datos del paciente", "en": "Patient data"},
    "risk.step2": {"es": "Resultado", "en": "Result"},
    "risk.sec_personal": {"es": "Datos personales", "en": "Personal data"},
    "risk.sec_body": {"es": "Medidas corporales", "en": "Body measurements"},
    "risk.sec_bp": {"es": "Presión arterial (mmHg)", "en": "Blood pressure (mmHg)"},
    "risk.sec_lab": {"es": "Exámenes de laboratorio", "en": "Lab results"},
    "risk.sec_lab_sub": {"es": "Según el reporte del laboratorio.", "en": "As reported by the lab."},
    "risk.sec_habits": {"es": "Hábitos", "en": "Habits"},
    "risk.bmi_live": {"es": "IMC {bmi:.1f} · {cat}", "en": "BMI {bmi:.1f} · {cat}"},
    "risk.bp_live": {"es": "{hi}/{lo} · {stage}", "en": "{hi}/{lo} · {stage}"},
    "risk.submit": {"es": "Ver resultado", "en": "See result"},
    "risk.fix_first": {"es": "Corrige los datos marcados para continuar.", "en": "Fix the highlighted data to continue."},
    "risk.error.range": {"es": "{field} debe estar entre {lo} y {hi}.", "en": "{field} must be between {lo} and {hi}."},
    "risk.error.bp_order": {"es": "La presión sistólica (la alta) debe ser mayor que la diastólica (la baja).",
                            "en": "The systolic pressure (upper) must be higher than the diastolic (lower)."},
    "risk.error.bmi": {"es": "El IMC ({bmi}) no es plausible. Revisa la estatura y el peso.",
                       "en": "The BMI ({bmi}) is not plausible. Check height and weight."},
    "risk.error.missing": {"es": "Falta el dato «{field}» o no es un número.",
                           "en": "«{field}» is empty or is not a number."},
    "risk.error.category": {"es": "«{field}» solo puede tomar los valores {levels}.",
                            "en": "«{field}» can only take the values {levels}."},
    "risk.warning.age_extrapolation": {
        "es": "El modelo aprendió con personas de <b>40 a 65 años</b>. Para otras edades el resultado es menos confiable.",
        "en": "The model learned from people aged <b>40 to 65</b>. For other ages the result is less reliable."},
    "risk.tag": {"es": "Resultado de la evaluación", "en": "Assessment result"},
    "risk.level.low": {"es": "Riesgo bajo", "en": "Low risk"},
    "risk.level.mid": {"es": "Resultado no concluyente", "en": "Inconclusive result"},
    "risk.level.high": {"es": "Riesgo alto", "en": "High risk"},
    "risk.msg.low": {"es": "La probabilidad de enfermedad cardiovascular es baja. Mantener controles de rutina y hábitos saludables.",
                     "en": "The probability of cardiovascular disease is low. Keep routine check-ups and healthy habits."},
    "risk.msg.mid": {"es": "Los modelos no están seguros con este perfil. Se recomienda la valoración de un médico.",
                     "en": "The models are not sure about this profile. A doctor's review is recommended."},
    "risk.msg.high": {"es": "La probabilidad es alta. Se recomienda valoración médica y, si es posible, un electrocardiograma.",
                      "en": "The probability is high. A medical review and, if possible, an electrocardiogram are recommended."},
    "risk.big_sub": {
        "es": "Probabilidad estimada: de cada 100 personas con un perfil como este, unas <b>{n}</b> tienen enfermedad cardiovascular.",
        "en": "Estimated probability: out of 100 people with a profile like this one, about <b>{n}</b> have cardiovascular disease."},
    "risk.bar_low": {"es": "Bajo", "en": "Low"},
    "risk.bar_mid": {"es": "Dudoso", "en": "Uncertain"},
    "risk.bar_high": {"es": "Alto", "en": "High"},
    "risk.cta_ecg": {"es": "Analizar su electrocardiograma", "en": "Analyse their electrocardiogram"},
    "risk.tab_why": {"es": "¿Por qué este resultado?", "en": "Why this result?"},
    "risk.tab_sim": {"es": "Simular cambios", "en": "Simulate changes"},
    "risk.tab_tech": {"es": "Detalle técnico", "en": "Technical detail"},
    "risk.why_title": {"es": "Lo que más influye", "en": "What matters most"},
    "risk.why_sub": {"es": "Comparado con un paciente típico del estudio. Los puntos indican cuánto sube o baja la probabilidad.",
                     "en": "Compared with a typical patient of the study. Points show how much the probability goes up or down."},
    "risk.why_none": {"es": "Todos los factores están cerca de lo típico: ninguno cambia el resultado de forma importante.",
                      "en": "Every factor is close to typical: none changes the result in an important way."},
    "risk.why_rest": {"es": "Los demás factores están en valores típicos y casi no cambian el resultado.",
                      "en": "The other factors are at typical values and barely change the result."},
    "risk.group_title": {"es": "Pacientes con un perfil parecido", "en": "Patients with a similar profile"},
    "risk.group_text": {
        "es": "Por su edad, peso, presión y laboratorios, este paciente se parece al grupo de <b>riesgo {level}</b>. "
              "En ese grupo, <b>{n} de cada 100</b> pacientes tenían enfermedad cardiovascular.",
        "en": "By age, weight, blood pressure and lab results, this patient resembles the <b>{level}-risk</b> group. "
              "In that group, <b>{n} out of 100</b> patients had cardiovascular disease."},
    "risk.sim_intro": {"es": "Mueve los controles para ver cómo cambiaría el riesgo si mejoran (o empeoran) estos valores.",
                       "en": "Move the controls to see how the risk would change if these values improve (or worsen)."},
    "risk.sim_now": {"es": "Riesgo actual", "en": "Current risk"},
    "risk.sim_new": {"es": "Con los cambios", "en": "With the changes"},
    "risk.sim_curve": {"es": "Cómo cambia el riesgo según la presión sistólica", "en": "How the risk changes with systolic pressure"},
    "risk.sim_curve_sub": {"es": "Con los demás valores de la simulación fijos. El punto rojo es el valor elegido.",
                           "en": "With the other simulated values fixed. The red dot is the chosen value."},
    "risk.prob_axis": {"es": "Probabilidad", "en": "Probability"},
    "risk.tech_intro": {
        "es": "El sistema usa <b>cuatro modelos</b> de machine learning entrenados por separado. Un <b>meta-modelo</b> "
              "combina sus probabilidades con pesos aprendidos (stacking) y entrega una sola respuesta.",
        "en": "The system uses <b>four</b> machine-learning <b>models</b> trained separately. A <b>meta-model</b> combines "
              "their probabilities with learned weights (stacking) and gives one answer."},
    "risk.tech_votes": {"es": "{n} de 4 modelos votan por enfermedad cardiovascular.",
                        "en": "{n} of 4 models vote for cardiovascular disease."},
    "risk.tech_equation": {"es": "Ecuación del meta-modelo con los valores de este paciente",
                           "en": "Meta-model equation with this patient's values"},
    "risk.col_model": {"es": "Modelo", "en": "Model"},
    "risk.col_prob": {"es": "Probabilidad", "en": "Probability"},
    "risk.col_weight": {"es": "Peso β", "en": "Weight β"},
    "risk.col_contrib": {"es": "Aporte β·p", "en": "Contribution β·p"},
    "risk.referral_rule": {
        "es": "Regla de remisión: el caso se marca como no concluyente si la probabilidad final está entre 35 % y 65 % "
              "o si los modelos empatan 2 a 2.",
        "en": "Referral rule: the case is marked inconclusive when the final probability is between 35 % and 65 % or "
              "when the models split 2–2."},
    "risk.save_title": {"es": "Guardar o compartir", "en": "Save or share"},

    # factor names and value descriptions
    "factor.age": {"es": "Edad", "en": "Age"},
    "factor.sex": {"es": "Sexo", "en": "Sex"},
    "factor.body": {"es": "Peso y estatura", "en": "Weight and height"},
    "factor.blood_pressure": {"es": "Presión arterial", "en": "Blood pressure"},
    "factor.cholesterol": {"es": "Colesterol", "en": "Cholesterol"},
    "factor.glucose": {"es": "Glucosa", "en": "Glucose"},
    "factor.smoking": {"es": "Tabaco", "en": "Smoking"},
    "factor.alcohol": {"es": "Alcohol", "en": "Alcohol"},
    "factor.activity": {"es": "Actividad física", "en": "Physical activity"},
    "fval.age": {"es": "{v:.0f} años", "en": "{v:.0f} years"},
    "fval.smoke.1": {"es": "Fuma", "en": "Smokes"},
    "fval.smoke.0": {"es": "No fuma", "en": "Does not smoke"},
    "fval.alco.1": {"es": "Consume alcohol", "en": "Drinks alcohol"},
    "fval.alco.0": {"es": "No consume alcohol", "en": "Does not drink alcohol"},
    "fval.active.1": {"es": "Hace actividad física", "en": "Physically active"},
    "fval.active.0": {"es": "No hace actividad física", "en": "Not physically active"},

    # -------------------------------------------------------------------- ecg
    "ecg.title": {"es": "Analizar un electrocardiograma", "en": "Analyse an electrocardiogram"},
    "ecg.subtitle": {"es": "Revisa latido por latido el ECG de una derivación (Holter o dispositivo portátil).",
                     "en": "Review, beat by beat, a single-lead ECG (Holter or wearable device)."},
    "ecg.step1": {"es": "Elegir el ECG", "en": "Choose the ECG"},
    "ecg.step2": {"es": "Resultado", "en": "Result"},
    "ecg.source_q": {"es": "¿De dónde viene el ECG?", "en": "Where does the ECG come from?"},
    "ecg.src.demo": {"es": "Usar un ejemplo", "en": "Use an example"},
    "ecg.src.csv": {"es": "Subir archivo CSV", "en": "Upload a CSV file"},
    "ecg.src.wfdb": {"es": "Subir archivo WFDB", "en": "Upload a WFDB file"},
    "ecg.demo_pick": {"es": "Elige un ejemplo real (PhysioNet)", "en": "Pick a real example (PhysioNet)"},
    "ecg.demo_labels": {"es": "Según la anotación experta: {n} normales y {a} anormales.",
                        "en": "According to the expert annotation: {n} normal and {a} abnormal."},
    "ecg.no_demos": {"es": "No se encontraron los ejemplos en sample_data/ecg.", "en": "Examples not found in sample_data/ecg."},
    "ecg.csv_upload": {"es": "Archivo CSV con la señal", "en": "CSV file with the signal"},
    "ecg.csv_format": {"es": "¿Qué formato necesito?", "en": "What format do I need?"},
    "ecg.csv_format_text": {
        "es": "Una fila por muestra y una columna numérica con el voltaje del ECG (en mV o µV). Puede incluir una "
              "columna de tiempo. Usa la derivación II si tu equipo graba varias. Se necesitan al menos 20 segundos.",
        "en": "One row per sample and one numeric column with the ECG voltage (mV or µV). A time column is optional. "
              "Use lead II if your device records several. At least 20 seconds are needed."},
    "ecg.csv_no_numeric": {"es": "El archivo no tiene columnas numéricas.", "en": "The file has no numeric columns."},
    "ecg.column": {"es": "Columna con la señal", "en": "Signal column"},
    "ecg.fs": {"es": "Muestras por segundo (Hz)", "en": "Samples per second (Hz)"},
    "ecg.fs_help": {"es": "La frecuencia de muestreo del equipo (los Holter suelen usar entre 125 y 500 Hz).",
                    "en": "The device's sampling rate (Holters usually use 125–500 Hz)."},
    "ecg.units": {"es": "Unidades", "en": "Units"},
    "ecg.wfdb_upload": {"es": "Archivos .hea y .dat del registro", "en": "The record's .hea and .dat files"},
    "ecg.lead": {"es": "Derivación", "en": "Lead"},
    "ecg.ready": {"es": "Señal lista: {s:.0f} segundos a {fs:.0f} Hz.", "en": "Signal ready: {s:.0f} seconds at {fs:.0f} Hz."},
    "ecg.waiting": {"es": "Elige un ejemplo o sube un archivo para continuar.", "en": "Pick an example or upload a file to continue."},
    "ecg.analyze": {"es": "Analizar ECG", "en": "Analyse ECG"},
    "ecg.running": {"es": "Buscando y clasificando los latidos…", "en": "Finding and classifying the heartbeats…"},
    "ecg.error.too_short": {"es": "La señal es muy corta: se necesitan al menos 20 segundos.",
                            "en": "The signal is too short: at least 20 seconds are needed."},
    "ecg.error.no_beats": {"es": "Se encontraron muy pocos latidos. Revisa la columna, las unidades y las muestras por segundo.",
                           "en": "Too few heartbeats were found. Check the column, the units and the samples per second."},
    "ecg.error.missing": {"es": "Más del 2 % de los valores de la señal están vacíos o no son números. Revisa la columna elegida.",
                          "en": "More than 2 % of the signal values are empty or not numbers. Check the chosen column."},
    "ecg.error.units_uv": {"es": "Los latidos miden más de 15 mV, algo imposible en un ECG: la señal parece estar en "
                                 "microvoltios. Cambia las unidades a µV.",
                           "en": "The beats measure more than 15 mV, which is impossible in an ECG: the signal seems to "
                                 "be in microvolts. Change the units to µV."},
    "ecg.error.units_small": {"es": "Los latidos miden menos de 0,03 mV, demasiado poco para un ECG: revisa las unidades "
                                    "(si el archivo está en milivoltios, elige mV).",
                              "en": "The beats measure less than 0.03 mV, too little for an ECG: check the units (if the "
                                    "file is in millivolts, choose mV)."},
    "ecg.error.heart_rate": {"es": "La frecuencia cardiaca calculada no es posible (menos de 30 o más de 220 latidos por "
                                   "minuto). Lo más probable es que las muestras por segundo no correspondan al archivo: "
                                   "revisa ese valor.",
                             "en": "The calculated heart rate is not possible (below 30 or above 220 beats per minute). "
                                   "Most likely the samples per second do not match the file: check that value."},
    "ecg.error.not_ecg": {"es": "La señal no parece un electrocardiograma: no tiene picos de latido marcados y las formas "
                                "encontradas no se parecen entre sí. Revisa que elegiste la columna correcta, las unidades "
                                "y las muestras por segundo.",
                          "en": "The signal does not look like an electrocardiogram: it has no clear heartbeat peaks and the "
                                "shapes found do not look alike. Check that you chose the right column, the units and the "
                                "samples per second."},
    "ecg.back": {"es": "Elegir otro ECG", "en": "Choose another ECG"},
    "ecg.tag": {"es": "Resultado del electrocardiograma", "en": "Electrocardiogram result"},
    "ecg.level.low": {"es": "Sin arritmias relevantes", "en": "No relevant arrhythmia"},
    "ecg.level.mid": {"es": "Latidos anormales ocasionales", "en": "Occasional abnormal beats"},
    "ecg.level.high": {"es": "Latidos anormales frecuentes", "en": "Frequent abnormal beats"},
    "ecg.msg.low": {"es": "Menos del 1 % de los latidos son anormales. No se observa una alteración importante del ritmo.",
                    "en": "Less than 1 % of the beats are abnormal. No important rhythm disturbance is seen."},
    "ecg.msg.mid": {"es": "Entre el 1 % y el 10 % de los latidos son anormales. Se sugiere seguimiento médico.",
                    "en": "Between 1 % and 10 % of the beats are abnormal. Medical follow-up is suggested."},
    "ecg.msg.high": {"es": "Más del 10 % de los latidos son anormales. Se recomienda valoración por cardiología.",
                     "en": "More than 10 % of the beats are abnormal. A cardiology review is recommended."},
    "ecg.big_sub": {"es": "<b>{a}</b> de los <b>{n}</b> latidos analizados son anormales.",
                    "en": "<b>{a}</b> of the <b>{n}</b> analysed beats are abnormal."},
    "ecg.guide_result_what": {
        "es": "El porcentaje grande es la <b>carga de latidos anormales</b>: de todos los latidos clasificados, cuántos "
              "salieron anormales (supraventriculares, ventriculares o de fusión).",
        "en": "The big percentage is the <b>abnormal-beat burden</b>: of all classified beats, how many came out abnormal "
              "(supraventricular, ventricular or fusion)."},
    "ecg.guide_result_read": {
        "es": "Verde: menos de 1 % (habitual en personas sanas). Amarillo: 1–10 %, conviene seguimiento. Rojo: más de 10 %, "
              "se recomienda valoración por cardiología. Abajo, el visor muestra dónde está cada latido anormal.",
        "en": "Green: under 1 % (common in healthy people). Yellow: 1–10 %, follow-up advised. Red: over 10 %, a "
              "cardiology review is recommended. Below, the viewer shows where each abnormal beat is."},
    "ecg.stat_hr": {"es": "Frecuencia cardiaca", "en": "Heart rate"},
    "ecg.stat_hr_v": {"es": "{v:.0f} lpm", "en": "{v:.0f} bpm"},
    "ecg.stat_hr_h": {"es": "entre {lo:.0f} y {hi:.0f} lpm", "en": "between {lo:.0f} and {hi:.0f} bpm"},
    "ecg.stat_dur": {"es": "Duración", "en": "Duration"},
    "ecg.stat_dur_v": {"es": "{m:.1f} min", "en": "{m:.1f} min"},
    "ecg.stat_beats": {"es": "Latidos analizados", "en": "Beats analysed"},
    "ecg.stat_run": {"es": "Racha anormal más larga", "en": "Longest abnormal run"},
    "ecg.stat_run_h": {"es": "latidos seguidos", "en": "beats in a row"},
    "ecg.viewer_title": {"es": "Electrocardiograma", "en": "Electrocardiogram"},
    "ecg.viewer_sub": {"es": "Cada punto es un latido. Pasa el mouse por encima para ver su probabilidad.",
                       "en": "Each dot is a heartbeat. Hover over it to see its probability."},
    "ecg.window": {"es": "Mostrar", "en": "Show"},
    "ecg.seconds": {"es": "{n} s", "en": "{n} s"},
    "ecg.start": {"es": "Desde el segundo", "en": "From second"},
    "ecg.prev_abn": {"es": "Anormal anterior", "en": "Previous abnormal"},
    "ecg.next_abn": {"es": "Siguiente anormal", "en": "Next abnormal"},
    "ecg.normal": {"es": "Latido normal", "en": "Normal beat"},
    "ecg.abnormal": {"es": "Latido anormal", "en": "Abnormal beat"},
    "ecg.unclass": {"es": "No clasificable", "en": "Unclassifiable"},
    "ecg.hover": {"es": "Segundo %{x:.1f}<br>Probabilidad de anormal: %{customdata[0]:.0%}",
                  "en": "Second %{x:.1f}<br>Probability of abnormal: %{customdata[0]:.0%}"},
    "ecg.time_s": {"es": "Tiempo (segundos)", "en": "Time (seconds)"},
    "ecg.exp_beat": {"es": "Ver un latido de cerca", "en": "Look at one beat closely"},
    "ecg.beat_pick": {"es": "Elige el latido", "en": "Choose the beat"},
    "ecg.beat_label": {"es": "Segundo {t:.1f} · {status}", "en": "Second {t:.1f} · {status}"},
    "ecg.beat_shape": {"es": "Forma del latido", "en": "Beat shape"},
    "ecg.beat_shape_sub": {"es": "Comparado con el latido habitual de este paciente.", "en": "Compared with this patient's usual beat."},
    "ecg.template": {"es": "Latido habitual", "en": "Usual beat"},
    "ecg.this_beat": {"es": "Este latido", "en": "This beat"},
    "ecg.time_ms": {"es": "Milisegundos desde el pico", "en": "Milliseconds from the peak"},
    "ecg.models_title": {"es": "Opinión de cada modelo", "en": "Each model's opinion"},
    "ecg.models_sub": {"es": "Probabilidad de que el latido sea anormal.", "en": "Probability that the beat is abnormal."},
    "ecg.beat_reason": {
        "es": "Llegó <b>{pct:.0f} %</b> {when} de lo esperado y su forma se parece un <b>{corr:.0f} %</b> al latido habitual.",
        "en": "It arrived <b>{pct:.0f} %</b> {when} than expected and its shape is <b>{corr:.0f} %</b> similar to the usual beat."},
    "ecg.beat_reason_ontime": {
        "es": "Llegó a tiempo y su forma se parece un <b>{corr:.0f} %</b> al latido habitual.",
        "en": "It arrived on time and its shape is <b>{corr:.0f} %</b> similar to the usual beat."},
    "ecg.earlier": {"es": "antes", "en": "earlier"},
    "ecg.later": {"es": "después", "en": "later"},
    "ecg.exp_rhythm": {"es": "Ritmo durante todo el registro", "en": "Rhythm over the whole recording"},
    "ecg.rhythm_sub": {
        "es": "Tiempo entre un latido y el anterior. Los latidos prematuros aparecen por debajo de la franja normal.",
        "en": "Time between each beat and the previous one. Premature beats appear below the normal band."},
    "ecg.rr_ms": {"es": "Tiempo desde el latido anterior (ms)", "en": "Time since previous beat (ms)"},
    "ecg.exp_table": {"es": "Tabla de todos los latidos", "en": "Table of every beat"},
    "ecg.col_time": {"es": "Segundo", "en": "Second"},
    "ecg.col_rr": {"es": "Desde el anterior (ms)", "en": "Since previous (ms)"},
    "ecg.col_p": {"es": "Probabilidad de anormal", "en": "Probability of abnormal"},
    "ecg.col_status": {"es": "Clasificación", "en": "Classification"},
    "ecg.exp_tech": {"es": "Detalles técnicos", "en": "Technical details"},
    "ecg.tech_text": {
        "es": "<b>Variabilidad de la frecuencia</b> (latidos normales): SDNN {sdnn:.0f} ms · RMSSD {rmssd:.0f} ms.<br>"
              "<b>Latidos no clasificables:</b> {unc} (intervalo imposible, por ruido o latidos no detectados).<br>"
              "<b>Latidos dudosos:</b> {ref} (probabilidad entre 35 % y 65 % o empate 2 a 2 de los modelos).<br>"
              "<b>Método:</b> filtro 0,5–40 Hz, detección automática de latidos (XQRS), 100 características por latido "
              "y clasificación con 4 modelos combinados por stacking.<br><b>Exactitud:</b> {known:.1%} en los latidos de "
              "prueba (protocolo estándar MIT-BIH). Análisis adicional con pacientes totalmente nuevos: {new:.1%} ± {sd:.1%}.",
        "en": "<b>Heart-rate variability</b> (normal beats): SDNN {sdnn:.0f} ms · RMSSD {rmssd:.0f} ms.<br>"
              "<b>Unclassifiable beats:</b> {unc} (impossible interval, due to noise or missed beats).<br>"
              "<b>Uncertain beats:</b> {ref} (probability between 35 % and 65 % or a 2–2 model split).<br>"
              "<b>Method:</b> 0.5–40 Hz filter, automatic beat detection (XQRS), 100 features per beat and "
              "classification with 4 models combined by stacking.<br><b>Accuracy:</b> {known:.1%} on the test beats "
              "(standard MIT-BIH protocol). Additional analysis on completely new patients: {new:.1%} ± {sd:.1%}."},

    # ---------------------------------------------------------------- history
    "hist.title": {"es": "Historial de pacientes", "en": "Patient history"},
    "hist.subtitle": {"es": "Evaluaciones guardadas en este equipo.", "en": "Assessments saved on this computer."},
    "hist.empty_title": {"es": "Aún no hay evaluaciones guardadas", "en": "No saved assessments yet"},
    "hist.empty_text": {"es": "Cuando evalúes a un paciente o analices un ECG, usa el botón «Guardar en el historial».",
                        "en": "After assessing a patient or analysing an ECG, use the 'Save to history' button."},
    "hist.search": {"es": "Buscar por código de paciente", "en": "Search by patient code"},
    "hist.filter": {"es": "Tipo", "en": "Type"},
    "hist.all": {"es": "Todos", "en": "All"},
    "hist.type_clinical": {"es": "Riesgo", "en": "Risk"},
    "hist.type_ecg": {"es": "ECG", "en": "ECG"},
    "hist.count": {"es": "{n} registro(s)", "en": "{n} record(s)"},
    "hist.col_date": {"es": "Fecha", "en": "Date"},
    "hist.col_type": {"es": "Tipo", "en": "Type"},
    "hist.col_patient": {"es": "Paciente", "en": "Patient"},
    "hist.col_result": {"es": "Resultado", "en": "Result"},
    "hist.no_code": {"es": "Sin código", "en": "No code"},
    "hist.res_clinical": {"es": "{level} · {p:.0%}", "en": "{level} · {p:.0%}"},
    "hist.res_ecg": {"es": "{level} · {p:.0%} anormales", "en": "{level} · {p:.0%} abnormal"},
    "hist.select_hint": {"es": "Selecciona una fila para ver el detalle o borrarla.",
                         "en": "Select a row to see its details or delete it."},
    "hist.detail": {"es": "Registro #{id} · {date}", "en": "Record #{id} · {date}"},
    "hist.inputs": {"es": "Datos ingresados", "en": "Entered data"},
    "hist.raw": {"es": "Ver todos los datos guardados", "en": "See all the saved data"},
    "hist.delete": {"es": "Borrar seleccionados ({n})", "en": "Delete selected ({n})"},
    "hist.confirm_title": {"es": "¿Borrar registros?", "en": "Delete records?"},
    "hist.confirm_text": {"es": "Se borrarán {n} registro(s) de forma permanente.", "en": "{n} record(s) will be deleted permanently."},
    "hist.confirm_yes": {"es": "Sí, borrar", "en": "Yes, delete"},
    "hist.cancel": {"es": "Cancelar", "en": "Cancel"},
    "hist.export": {"es": "Exportar historial (CSV)", "en": "Export history (CSV)"},

    # ------------------------------------------------------------------ batch
    "batch.title": {"es": "Evaluar varios pacientes", "en": "Assess several patients"},
    "batch.subtitle": {"es": "Sube un archivo con muchos pacientes y obtén el resultado de todos a la vez.",
                       "en": "Upload a file with many patients and get everyone's result at once."},
    "batch.step1": {"es": "Preparar el archivo", "en": "Prepare the file"},
    "batch.step2": {"es": "Subir el archivo", "en": "Upload the file"},
    "batch.step3": {"es": "Resultados", "en": "Results"},
    "batch.step1_text": {"es": "Descarga la plantilla, llena una fila por paciente y guárdala como CSV.",
                         "en": "Download the template, fill in one row per patient and save it as CSV."},
    "batch.template": {"es": "Descargar plantilla", "en": "Download template"},
    "batch.columns": {"es": "Ver qué significa cada columna", "en": "What each column means"},
    "batch.columns_help": {
        "es": "`age_years` edad · `gender` 1 mujer, 2 hombre · `height` cm · `weight` kg · `ap_hi`, `ap_lo` presión "
              "sistólica y diastólica · `cholesterol`, `gluc` 1 normal, 2 alto, 3 muy alto · `smoke`, `alco`, `active` "
              "0 no, 1 sí. Opcional: `patient_code` y `cardio` (diagnóstico real, 0/1, para comparar).",
        "en": "`age_years` age · `gender` 1 female, 2 male · `height` cm · `weight` kg · `ap_hi`, `ap_lo` systolic and "
              "diastolic pressure · `cholesterol`, `gluc` 1 normal, 2 high, 3 very high · `smoke`, `alco`, `active` "
              "0 no, 1 yes. Optional: `patient_code` and `cardio` (true diagnosis, 0/1, to compare)."},
    "batch.upload": {"es": "Archivo CSV de pacientes", "en": "Patients CSV file"},
    "batch.example": {"es": "Probar con 300 pacientes de ejemplo", "en": "Try with 300 example patients"},
    "batch.example_help": {"es": "Pacientes reales del conjunto de prueba, que el modelo no usó para aprender.",
                           "en": "Real patients from the test set, not used by the model to learn."},
    "batch.missing": {"es": "Al archivo le faltan estas columnas: {cols}", "en": "The file is missing these columns: {cols}"},
    "batch.empty": {"es": "El archivo no tiene filas de pacientes.", "en": "The file has no patient rows."},
    "batch.summary": {"es": "Se evaluaron <b>{n}</b> pacientes.", "en": "<b>{n}</b> patients were assessed."},
    "batch.rejected": {"es": "{n} fila(s) no se evaluaron por tener datos faltantes o imposibles",
                      "en": "{n} row(s) were not assessed because of missing or impossible data"},
    "batch.running": {"es": "Evaluando pacientes…", "en": "Assessing patients…"},
    "batch.col_patient": {"es": "Paciente", "en": "Patient"},
    "batch.col_age": {"es": "Edad", "en": "Age"},
    "batch.col_bp": {"es": "Presión", "en": "Blood pressure"},
    "batch.col_bmi": {"es": "IMC", "en": "BMI"},
    "batch.col_prob": {"es": "Probabilidad", "en": "Probability"},
    "batch.col_result": {"es": "Resultado", "en": "Result"},
    "batch.download": {"es": "Descargar resultados (CSV)", "en": "Download results (CSV)"},
    "batch.compare": {"es": "Comparar con el diagnóstico real", "en": "Compare with the true diagnosis"},
    "batch.compare_text": {
        "es": "El archivo trae el diagnóstico real (columna <code>cardio</code>). El sistema acertó en <b>{acc:.0%}</b> de los "
              "pacientes: detectó <b>{sens:.0%}</b> de los enfermos y descartó correctamente a <b>{spec:.0%}</b> de los sanos.",
        "en": "The file includes the true diagnosis (<code>cardio</code> column). The system was right for <b>{acc:.0%}</b> "
              "of the patients: it caught <b>{sens:.0%}</b> of the sick ones and correctly cleared <b>{spec:.0%}</b> of the "
              "healthy ones."},

    # ------------------------------------------------------------ how it works
    "how.title": {"es": "Cómo funciona", "en": "How it works"},
    "how.subtitle": {"es": "Qué hace el sistema, qué tan bien funciona y de dónde vienen los datos.",
                     "en": "What the system does, how well it works and where the data comes from."},
    "how.tab_simple": {"es": "En pocas palabras", "en": "In a nutshell"},
    "how.tab_results": {"es": "Resultados", "en": "Results"},
    "how.tab_trace": {"es": "Gráficas y trazabilidad", "en": "Figures and traceability"},
    "how.tab_features": {"es": "Funcionalidades", "en": "Functionalities"},
    "how.tab_sources": {"es": "Fuentes", "en": "Sources"},
    "how.tier1_title": {"es": "Nivel 1 · Tamizaje de riesgo", "en": "Tier 1 · Risk screening"},
    "how.tier1_q": {"es": "¿Este paciente probablemente tiene enfermedad cardiovascular?",
                    "en": "Does this patient probably have cardiovascular disease?"},
    "how.tier1_data": {"es": "Aprendió de un examen básico de <b>{n:,}</b> pacientes (dataset Cardiovascular Disease, Kaggle).",
                       "en": "Learned from a basic exam of <b>{n:,}</b> patients (Cardiovascular Disease dataset, Kaggle)."},
    "how.tier2_title": {"es": "Nivel 2 · Diagnóstico por ECG", "en": "Tier 2 · ECG diagnosis"},
    "how.tier2_q": {"es": "¿Este latido del corazón es normal o anormal?", "en": "Is this heartbeat normal or abnormal?"},
    "how.tier2_data": {"es": "Aprendió de <b>{n:,}</b> latidos anotados por expertos (bases MIT-BIH, PhysioNet).",
                       "en": "Learned from <b>{n:,}</b> expert-annotated beats (MIT-BIH databases, PhysioNet)."},
    "how.acc": {"es": "Acierta en <b>{acc:.0%}</b> de los casos de prueba.", "en": "Right in <b>{acc:.0%}</b> of the test cases."},
    "how.flow_title": {"es": "Cómo trabajan juntos", "en": "How they work together"},
    "how.flow_text": {
        "es": "El nivel 1 es un filtro rápido: solo necesita un examen de rutina. Si el riesgo es alto o dudoso, el "
              "nivel 2 revisa el electrocardiograma para buscar latidos anormales. En cada nivel, cuatro modelos dan su "
              "opinión por separado y un modelo final las combina en una sola respuesta.",
        "en": "Tier 1 is a quick filter: it only needs a routine exam. If the risk is high or uncertain, tier 2 checks "
              "the electrocardiogram for abnormal beats. In each tier, four models give their opinion separately and a "
              "final model combines them into one answer."},
    "how.dot_input": {"es": "Datos del paciente", "en": "Patient data"},
    "how.dot_meta": {"es": "Modelo final", "en": "Final model"},
    "how.dot_answer": {"es": "Una respuesta\\n(o remisión si hay duda)", "en": "One answer\\n(or referral if unsure)"},
    "how.limits_title": {"es": "Límites que debes conocer", "en": "Limits you should know"},
    "how.limits": {
        "es": "- El nivel 1 acierta en ~73 % de los casos: es el máximo que permiten esos 11 datos. Úsalo como filtro, no como diagnóstico.\n"
              "- El nivel 1 aprendió con personas de 40 a 65 años; fuera de ese rango es menos confiable.\n"
              "- En estos datos, quienes fuman, toman alcohol o tienen glucosa muy alta aparecen con *menos* enfermedad "
              "(un sesgo conocido del dataset). Por eso los modelos siguen reglas clínicas: el riesgo nunca baja cuando "
              "sube la presión, la edad, el peso, el colesterol, la glucosa, el tabaco o el alcohol.\n"
              "- El nivel 2 acierta en {known:.1%} de los latidos de prueba (protocolo estándar MIT-BIH). Con pacientes "
              "totalmente nuevos (análisis adicional por paciente) baja a {new:.1%} ± {sd:.1%}; los latidos "
              "supraventriculares y de fusión son los más difíciles.\n"
              "- El nivel 2 necesita al menos 20 segundos de una derivación del ECG. Los ECG de ejemplo pertenecen a las "
              "bases con las que se entrenó el modelo.\n"
              "- Es un prototipo académico: no es un dispositivo médico certificado.",
        "en": "- Tier 1 is right in ~73 % of the cases: the most those 11 fields allow. Use it as a filter, not a diagnosis.\n"
              "- Tier 1 learned from people aged 40 to 65; outside that range it is less reliable.\n"
              "- In this data, smokers, drinkers and people with very high glucose show *less* disease (a known bias of "
              "the dataset). That is why the models follow clinical rules: the risk never goes down when blood pressure, "
              "age, weight, cholesterol, glucose, smoking or alcohol go up.\n"
              "- Tier 2 is right in {known:.1%} of the test beats (standard MIT-BIH protocol). On completely new patients "
              "(additional patient-wise analysis) it drops to {new:.1%} ± {sd:.1%}; supraventricular and fusion beats "
              "are the hardest.\n"
              "- Tier 2 needs at least 20 seconds of one ECG lead. The example ECGs belong to the databases the model was "
              "trained on.\n"
              "- It is an academic prototype: not a certified medical device."},
    "how.pick_tier": {"es": "Nivel", "en": "Tier"},
    "how.tier1_short": {"es": "Nivel 1 · Riesgo clínico", "en": "Tier 1 · Clinical risk"},
    "how.tier2_short": {"es": "Nivel 2 · ECG", "en": "Tier 2 · ECG"},
    "how.records": {"es": "Registros", "en": "Records"},
    "how.records_h": {"es": "balanceados 50/50", "en": "balanced 50/50"},
    "how.accuracy": {"es": "Exactitud", "en": "Accuracy"},
    "how.accuracy_h": {"es": "IC 95 %: {lo:.1%} – {hi:.1%}", "en": "95 % CI: {lo:.1%} – {hi:.1%}"},
    "how.auc_h": {"es": "capacidad de separar clases", "en": "ability to separate classes"},
    "how.target": {"es": "Meta de la rúbrica (> 95 %)", "en": "Rubric target (> 95 %)"},
    "how.target_ok": {"es": "Cumple", "en": "Met"},
    "how.target_ok_h": {"es": "protocolo estándar MIT-BIH", "en": "standard MIT-BIH protocol"},
    "how.chart_title": {"es": "Cada modelo frente a la respuesta combinada", "en": "Each model vs. the combined answer"},
    "how.chart_sub": {"es": "Exactitud en el conjunto de prueba (datos que el sistema no vio al aprender).",
                      "en": "Accuracy on the test set (data the system did not see while learning)."},
    "how.table": {"es": "Ver todas las métricas", "en": "See every metric"},
    "how.equation": {"es": "Ecuación aprendida por el modelo final", "en": "Equation learned by the final model"},
    "how.tab_glossary": {"es": "Glosario", "en": "Glossary"},
    "how.acc_ecg": {"es": "Acierta en <b>{acc}</b> de los latidos de prueba: supera la meta del 95 %.",
                   "en": "Right in <b>{acc}</b> of the test beats: above the 95 % target."},
    "how.target_no_text": {
        "es": "Este nivel no alcanza el 95 %: con 11 datos de un examen básico el máximo posible ronda el 73 %. Por eso "
              "existe el nivel 2 (ECG), que aporta la señal del corazón.",
        "en": "This tier does not reach 95 %: with 11 fields of a basic exam the ceiling is about 73 %. That is why tier 2 "
              "(ECG) exists: it adds the heart's own signal."},
    "how.protocols_title": {"es": "Análisis adicional: pacientes totalmente nuevos",
                            "en": "Additional analysis: completely new patients"},
    "how.protocols_sub": {"es": "La cifra oficial usa el protocolo estándar de MIT-BIH. Como prueba extra de robustez, "
                                "el sistema también se evaluó con pacientes que nunca vio.",
                          "en": "The official figure uses the standard MIT-BIH protocol. As an extra robustness test, the "
                                "system was also evaluated on patients it never saw."},
    "how.known_title": {"es": "Oficial · división por latido", "en": "Official · split by beat"},
    "how.known_text": {"es": "Protocolo estándar de los artículos sobre MIT-BIH (intra-paciente): los latidos de prueba "
                             "nunca se usaron para entrenar. Es la cifra que cumple la meta del 95 %.",
                       "en": "Standard protocol of MIT-BIH papers (intra-patient): the test beats were never used for "
                             "training. It is the figure that meets the 95 % target."},
    "how.new_title": {"es": "Adicional · pacientes nuevos", "en": "Additional · new patients"},
    "how.new_text": {"es": "Validación cruzada de 5 particiones donde ningún paciente está a la vez en entrenamiento y "
                           "prueba (inter-paciente). Es más estricta y, como en la literatura, da una cifra menor.",
                     "en": "5-fold cross-validation where no patient is in training and test at the same time "
                           "(inter-patient). It is stricter and, as in the literature, gives a lower figure."},
    "how.folds_title": {"es": "Exactitud en cada partición por paciente", "en": "Accuracy in each patient-wise fold"},
    "how.leak_text": {
        "es": "<b>¿Por qué difieren?</b> En la división por latido, el <b>{overlap:.0%}</b> de los latidos de prueba son de "
              "pacientes que también están en el entrenamiento y, para el <b>{nn:.0%}</b>, el latido más parecido del "
              "entrenamiento es del mismo paciente. No es un error de datos: es parte del protocolo estándar, y por eso con "
              "una persona totalmente nueva la exactitud es menor.",
        "en": "<b>Why do they differ?</b> In the split by beat, <b>{overlap:.0%}</b> of the test beats come from patients "
              "who are also in training and, for <b>{nn:.0%}</b> of them, the most similar training beat is from the same "
              "patient. It is not a data error: it is part of the standard protocol, which is why accuracy is lower on a "
              "completely new person."},
    "how.class_recall": {
        "es": "Con pacientes nuevos reconoce el {N:.0%} de los latidos normales, el {V:.0%} de los ventriculares, el "
              "{S:.0%} de los supraventriculares y el {F:.0%} de los de fusión.",
        "en": "On new patients it recognises {N:.0%} of normal beats, {V:.0%} of ventricular, {S:.0%} of "
              "supraventricular and {F:.0%} of fusion beats."},
    "how.auto_text": {
        "es": "Prueba de despliegue: con latidos detectados automáticamente (como en la app), el detector encontró el "
              "{det:.1%} de los latidos y la exactitud balanceada fue {bal:.1%} en pacientes nuevos.",
        "en": "Deployment check: with automatically detected beats (as in the app), the detector found {det:.1%} of the "
              "beats and the balanced accuracy was {bal:.1%} on new patients."},
    "how.vote_hard": {"es": "Votación simple", "en": "Majority vote"},
    "how.vote_soft": {"es": "Promedio de probabilidades", "en": "Average of probabilities"},
    "how.chart_sub_ecg": {"es": "En el ECG: protocolo estándar MIT-BIH (división por latido).",
                          "en": "For the ECG: standard MIT-BIH protocol (split by beat)."},
    "how.equation_text": {
        "es": "σ es la función sigmoide, que convierte la suma en una probabilidad entre 0 y 1; cada p es la probabilidad "
              "que dio un modelo y su peso dice cuánto confía en él el modelo final.",
        "en": "σ is the sigmoid function, which turns the sum into a probability between 0 and 1; each p is one model's "
              "probability and its weight says how much the final model trusts it."},
    "how.figs_intro": {"es": "Todas las gráficas que generó el entrenamiento, agrupadas por tema. Al lado de cada una "
                             "está qué muestra y cómo leerla.",
                       "en": "Every figure produced by training, grouped by topic. Next to each one: what it shows and "
                             "how to read it."},
    "how.category": {"es": "Tema", "en": "Topic"},
    "how.cat.data": {"es": "Datos y limpieza", "en": "Data and cleaning"},
    "how.cat.eda": {"es": "Exploración (EDA)", "en": "Exploration (EDA)"},
    "how.cat.models": {"es": "Desempeño de los modelos", "en": "Model performance"},
    "how.cat.explain": {"es": "Explicabilidad y grupos", "en": "Explainability and groups"},
    "how.cat.validation": {"es": "Validación adicional", "en": "Additional validation"},
    "how.cat.simulation": {"es": "Simulación", "en": "Simulation"},
    "how.glossary_intro": {"es": "Los términos que aparecen en la aplicación, explicados sin tecnicismos.",
                           "en": "The terms used in the application, explained without jargon."},
    "how.fig_pick": {"es": "Gráfica del entrenamiento", "en": "Training figure"},
    "how.lineage": {"es": "Limpieza de datos: filas eliminadas en cada paso", "en": "Data cleaning: rows removed at each step"},
    "how.trace_full": {"es": "Reporte completo de trazabilidad", "en": "Full traceability report"},
    "how.summary_full": {"es": "Resumen de resultados y rúbrica", "en": "Results summary and rubric"},
    "how.no_summary": {"es": "Ejecuta `python -m cardio_ml.report` en ml-pipeline para generar el resumen.",
                       "en": "Run `python -m cardio_ml.report` in ml-pipeline to build the summary."},
    "how.features_intro": {"es": "Las funcionalidades de la aplicación y dónde encontrarlas:",
                           "en": "The application's functionalities and where to find them:"},
    "how.authors": {"es": "Autores", "en": "Authors"},
    "feature.1": {"es": "Evaluación de riesgo de un paciente", "en": "Patient risk assessment"},
    "feature.2": {"es": "Detalle de cómo decidieron los 4 modelos", "en": "How the 4 models decided"},
    "feature.3": {"es": "Grupo de pacientes parecidos (K-means)", "en": "Similar-patient group (K-means)"},
    "feature.4": {"es": "Explicación de los factores que influyen", "en": "Explanation of the influencing factors"},
    "feature.5": {"es": "Simulador de cambios", "en": "Change simulator"},
    "feature.6": {"es": "Análisis de ECG latido por latido", "en": "Beat-by-beat ECG analysis"},
    "feature.7": {"es": "Visor de ECG e inspector de latidos", "en": "ECG viewer and beat inspector"},
    "feature.8": {"es": "Evaluación de varios pacientes (CSV)", "en": "Several-patient assessment (CSV)"},
    "feature.9": {"es": "Historial de pacientes", "en": "Patient history"},
    "feature.10": {"es": "Resultados y trazabilidad del modelo", "en": "Model results and traceability"},
    "feature.11": {"es": "Reporte PDF", "en": "PDF report"},
    "feature.12": {"es": "Interfaz bilingüe (español / inglés)", "en": "Bilingual interface (Spanish / English)"},

    # ------------------------------------------------------------------ files
    "files.wrong_csv": {"es": "«{name}» no es un archivo CSV. Sube un archivo que termine en .csv o .txt.",
                        "en": "'{name}' is not a CSV file. Upload a file ending in .csv or .txt."},
    "files.need_hea": {"es": "Falta el archivo de cabecera (.hea). Selecciona el .hea y el .dat del registro al mismo tiempo.",
                       "en": "The header file (.hea) is missing. Select the record's .hea and .dat files together."},
    "files.need_dat": {"es": "Falta el archivo «{name}.dat». Selecciona el .hea y el .dat al mismo tiempo.",
                       "en": "The file '{name}.dat' is missing. Select the .hea and the .dat together."},
    "files.unreadable": {"es": "No se pudo leer el archivo: {error}", "en": "The file could not be read: {error}"},

    # -------------------------------------------------------------------- pdf
    "pdf.clinical_title": {"es": "Evaluación de riesgo cardiovascular", "en": "Cardiovascular risk assessment"},
    "pdf.ecg_title": {"es": "Análisis de electrocardiograma", "en": "Electrocardiogram analysis"},
    "pdf.patient": {"es": "Datos del paciente", "en": "Patient data"},
    "pdf.result": {"es": "Resultado", "en": "Result"},
    "pdf.recommendation": {"es": "Recomendación", "en": "Recommendation"},
    "pdf.factors": {"es": "Factores que más influyen", "en": "Most influential factors"},
    "pdf.models": {"es": "Detalle técnico: los 4 modelos", "en": "Technical detail: the 4 models"},
    "pdf.probability": {"es": "Probabilidad estimada", "en": "Estimated probability"},
    "pdf.group": {"es": "Grupo de pacientes parecido", "en": "Similar-patient group"},

    # ---------------------------------------------------- reading guides & tour
    "guide.button": {"es": "¿Cómo leer esta gráfica?", "en": "How to read this chart?"},
    "guide.what": {"es": "Qué muestra", "en": "What it shows"},
    "guide.read": {"es": "Cómo leerla", "en": "How to read it"},
    "guide.result_button": {"es": "¿Cómo leer este resultado?", "en": "How to read this result?"},
    "guide.factors_button": {"es": "¿Qué significan los puntos?", "en": "What do the points mean?"},
    "chart.threshold": {"es": "Umbral 50 %", "en": "50 % threshold"},
    "chart.doubt_zone": {"es": "Zona de duda (35–65 %)", "en": "Doubt zone (35–65 %)"},
    "chart.your_patient": {"es": "Su paciente", "en": "Your patient"},
    "chart.risk_axis": {"es": "Probabilidad de ECV", "en": "Probability of CVD"},
    "chart.r_peak": {"es": "Pico R", "en": "R peak"},
    "chart.shape_axis": {"es": "Forma (normalizada)", "en": "Shape (normalised)"},
    "chart.target": {"es": "Meta 95 %", "en": "95 % target"},
    "chart.fold": {"es": "Partición {n}", "en": "Fold {n}"},
    "chart.mean": {"es": "Promedio", "en": "Mean"},
    "app.tour": {"es": "Guía rápida", "en": "Quick guide"},
    "tour.1_t": {"es": "Bienvenido a CardioAI", "en": "Welcome to CardioAI"},
    "tour.1": {"es": "<p>Un asistente de inteligencia artificial para apoyar la evaluación cardiovascular en telemedicina. "
                     "Tiene <b>dos herramientas</b>:</p><ul><li><b>Evaluar paciente</b>: riesgo de enfermedad cardiovascular "
                     "con 11 datos de un examen básico.</li><li><b>Analizar ECG</b>: revisa un electrocardiograma latido por "
                     "latido y marca los anormales.</li></ul>",
               "en": "<p>An artificial-intelligence assistant that supports cardiovascular assessment in telemedicine. "
                     "It has <b>two tools</b>:</p><ul><li><b>Assess a patient</b>: risk of cardiovascular disease from 11 "
                     "fields of a basic exam.</li><li><b>Analyse an ECG</b>: reviews an electrocardiogram beat by beat and "
                     "marks the abnormal ones.</li></ul>"},
    "tour.2_t": {"es": "Cuatro modelos, una respuesta", "en": "Four models, one answer"},
    "tour.2": {"es": "<p>Cada caso lo analizan <b>cuatro modelos de IA</b> por separado (regresión logística, vecinos "
                     "cercanos, bosque aleatorio y SVM). Un <b>modelo final</b> combina sus opiniones con pesos aprendidos "
                     "y entrega <b>un solo resultado</b>.</p><p>Si los modelos no se ponen de acuerdo, el sistema lo dice y "
                     "recomienda la revisión de un profesional.</p>",
               "en": "<p>Each case is analysed by <b>four AI models</b> separately (logistic regression, nearest "
                     "neighbours, random forest and SVM). A <b>final model</b> combines their opinions with learned weights "
                     "and gives <b>one result</b>.</p><p>If the models disagree, the system says so and recommends a "
                     "professional review.</p>"},
    "tour.3_t": {"es": "Cómo leer los resultados", "en": "How to read the results"},
    "tour.3": {"es": "<ul><li><b style='color:#16A34A'>Verde</b>: riesgo bajo o ritmo sin alteraciones.</li>"
                     "<li><b style='color:#D97706'>Amarillo</b>: zona de duda, conviene que lo revise un médico.</li>"
                     "<li><b style='color:#DC2626'>Rojo</b>: riesgo alto o latidos anormales frecuentes.</li></ul>"
                     "<p>Junto a cada gráfica hay un botón <b>«¿Cómo leer esta gráfica?»</b> que explica qué muestra.</p>",
               "en": "<ul><li><b style='color:#16A34A'>Green</b>: low risk or normal rhythm.</li>"
                     "<li><b style='color:#D97706'>Yellow</b>: doubt zone, a doctor should review it.</li>"
                     "<li><b style='color:#DC2626'>Red</b>: high risk or frequent abnormal beats.</li></ul>"
                     "<p>Next to each chart there is a <b>'How to read this chart?'</b> button that explains it.</p>"},
    "tour.4_t": {"es": "¿Por dónde empiezo?", "en": "Where do I start?"},
    "tour.4": {"es": "<p>Prueba <b>Evaluar paciente</b> con los datos que vienen por defecto, o abre <b>Analizar ECG</b> y "
                     "elige uno de los ejemplos reales. En <b>Cómo funciona</b> están los resultados, la trazabilidad y un "
                     "glosario de términos.</p><p>Puedes volver a abrir esta guía desde el menú lateral.</p>",
               "en": "<p>Try <b>Assess a patient</b> with the default data, or open <b>Analyse an ECG</b> and pick one of "
                     "the real examples. <b>How it works</b> has the results, the traceability and a glossary.</p><p>You can "
                     "reopen this guide from the side menu.</p>"},
    "tour.next": {"es": "Siguiente", "en": "Next"},
    "tour.back": {"es": "Atrás", "en": "Back"},
    "tour.skip": {"es": "Saltar guía", "en": "Skip guide"},
    "tour.start": {"es": "Empezar", "en": "Get started"},
    "home.eyebrow": {"es": "Telemedicina · Proyecto académico USCO", "en": "Telemedicine · USCO academic project"},
    "home.badge_models": {"es": "4 modelos → 1 respuesta", "en": "4 models → 1 answer"},
    "home.badge_data": {"es": "{n:,} pacientes y {b:,} latidos reales", "en": "{n:,} patients and {b:,} real beats"},
    "home.badge_lang": {"es": "Explicable y trazable", "en": "Explainable and traceable"},
    "home.start_here": {"es": "Elige una herramienta", "en": "Choose a tool"},
}

FIELDS = {"ap_hi": "in.ap_hi", "ap_lo": "in.ap_lo", "height": "in.height", "weight": "in.weight",
          "age_years": "in.age", "gender": "in.sex", "cholesterol": "in.cholesterol", "gluc": "in.glucose",
          "smoke": "in.smoke", "alco": "in.alcohol", "active": "in.active"}


def lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)


class _EsNumber(float):
    """Formats like Spanish: 67.772 · 98,5 % (thousands dot, decimal comma, spaced %)."""

    def __format__(self, spec: str) -> str:
        if not spec:                                   # "{n}" -> no decimals for whole numbers
            text = str(int(self)) if float(self).is_integer() else format(float(self), "g")
            return text.replace(".", ",")
        text = format(float(self), spec)
        if spec.endswith("%"):
            text = text[:-1] + " %"
        return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


class _EsInt(int):
    def __format__(self, spec: str) -> str:
        if spec.endswith("%") or "." in spec:
            return format(_EsNumber(self), spec)
        return format(int(self), spec).replace(",", ".")


def _localise(value):
    if lang() != "es" or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return _EsInt(value)
    if isinstance(value, float):
        return _EsNumber(value)
    if hasattr(value, "dtype"):                     # numpy scalars
        return _EsInt(int(value)) if value.dtype.kind in "iu" else _EsNumber(float(value))
    return value


def num(value: float, spec: str = ",.0f") -> str:
    """Format a number with the conventions of the current language."""
    return format(_localise(value), spec)


def t(key: str, **kwargs) -> str:
    entry = T.get(key)
    if entry is None:
        return key
    text = entry.get(lang(), entry["en"])
    return text.format(**{k: _localise(v) for k, v in kwargs.items()}) if kwargs else text


def level_label(value: int) -> str:
    return t(f"level.{int(value)}")


def validation_message(code: str) -> str:
    kind, *rest = code.split(":")
    if kind == "range":
        key, lo, hi = rest
        return t("risk.error.range", field=t(FIELDS[key]), lo=lo, hi=hi)
    if kind == "bmi":
        return t("risk.error.bmi", bmi=rest[0])
    if kind == "missing":
        return t("risk.error.missing", field=t(FIELDS[rest[0]]))
    if kind == "category":
        key, levels = rest
        return t("risk.error.category", field=t(FIELDS[key]), levels=levels.replace("/", ", "))
    return t("risk.error." + kind)
