# Technical report

- [Read the report](Cache_Aware_GNN_Distillation_Technical_Report.md)
- [Open or download the PDF](Cache_Aware_GNN_Distillation_Technical_Report.pdf)

Both copies contain the architecture, distillation and cache techniques, Amazon products example, and important results with and without FAITH. The PDF includes rendered diagrams and result tables.

The Markdown copy preserves the original report text, with relative evidence links adjusted for this folder. The original remains at [FAITH_CACHE_STUDY.md](../FAITH_CACHE_STUDY.md).

To rebuild the PDF, install `reportlab>=4` and `markdown-it-py>=3`, ensure DejaVu Sans fonts are available, then run `python tools/export_report_pdf.py` from the repository root.
