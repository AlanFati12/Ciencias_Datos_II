"""PDF report of an assessment (fpdf2, core fonts -> latin-1 text only)."""
from __future__ import annotations

from datetime import datetime

from fpdf import FPDF

REPLACEMENTS = {"–": "-", "—": "-", "≥": ">=", "≤": "<=", "β": "beta", "·": "-", "’": "'", "“": '"', "”": '"',
                "²": "2", "…": "..."}


def _latin1(text: str) -> str:
    for a, b in REPLACEMENTS.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    def __init__(self, title: str, footer_text: str) -> None:
        super().__init__()
        self.title_text, self.footer_text = _latin1(title), _latin1(footer_text)
        self.set_auto_page_break(True, margin=18)
        self.add_page()

    def header(self) -> None:
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(34, 34, 34)
        self.cell(0, 9, self.title_text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(110, 110, 110)
        self.cell(0, 5, datetime.now().strftime("%Y-%m-%d %H:%M"), new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.multi_cell(0, 3.5, self.footer_text, align="C")

    def section(self, text: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(46, 111, 158)
        self.cell(0, 7, _latin1(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(34, 34, 34)

    def table(self, rows: list[tuple[str, str]], widths=(80, 100)) -> None:
        self.set_font("Helvetica", "", 9.5)
        for i, (k, v) in enumerate(rows):
            self.set_fill_color(*((244, 246, 248) if i % 2 == 0 else (255, 255, 255)))
            self.cell(widths[0], 6.5, _latin1(str(k)), fill=True)
            self.cell(widths[1], 6.5, _latin1(str(v)), fill=True, new_x="LMARGIN", new_y="NEXT")

    def paragraph(self, text: str, bold: bool = False) -> None:
        self.set_font("Helvetica", "B" if bold else "", 10)
        self.multi_cell(0, 5.5, _latin1(text), new_x="LMARGIN", new_y="NEXT")


def build(title: str, sections: list[tuple[str, list[tuple[str, str]] | str]], footer: str) -> bytes:
    """sections: [(heading, rows or paragraph text), ...]"""
    pdf = _Report(title, footer)
    for heading, content in sections:
        pdf.section(heading)
        if isinstance(content, str):
            pdf.paragraph(content)
        else:
            pdf.table(content)
    return bytes(pdf.output())
