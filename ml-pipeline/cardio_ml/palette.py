"""Colour tokens shared by the figures and the models (no plotting imports, so the
web app can unpickle the models without matplotlib or seaborn)."""

# categorical slots (fixed order, never cycled)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"

CLASS_COLORS = {0: BLUE, 1: ORANGE}                    # negative / normal  vs.  positive / abnormal
MODEL_COLORS = {"LR": BLUE, "KNN": ORANGE, "RF": AQUA, "SVM": YELLOW}
ENSEMBLE_COLOR = INK
VOTE_COLORS = {"Soft voting": VIOLET, "Hard voting": MUTED}
RISK_COLORS = {"Low": GOOD, "Medium": WARNING, "High": CRITICAL}
AAMI_COLORS = {"N": BLUE, "S": ORANGE, "V": AQUA, "F": YELLOW}
