"""
Export analytics summary to PDF (bonus feature).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fpdf import FPDF

from utils.config import get_config


class AnalyticsPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Emotional Transition Analytics Report", border=False, ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def export_analytics_pdf(
    emotions: List[str],
    transitions: List[Dict],
    metrics: Optional[Dict] = None,
    save_path: Optional[Path] = None,
) -> Path:
    """Generate PDF report of conversation analytics."""
    config = get_config()
    save_path = save_path or config.outputs_dir / f"analytics_{datetime.now():%Y%m%d_%H%M%S}.pdf"

    pdf = AnalyticsPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Emotion Timeline", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for i, emo in enumerate(emotions, 1):
        pdf.cell(0, 6, f"  Step {i}: {emo}", ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detected Transitions", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for t in transitions:
        pdf.cell(0, 6, f"  {t.get('transition', 'N/A')} (changed={t.get('changed', False)})", ln=True)

    if metrics:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Model Metrics", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in metrics.items():
            if isinstance(v, (int, float, str)):
                pdf.cell(0, 6, f"  {k}: {v}", ln=True)

    pdf.output(str(save_path))
    return save_path
