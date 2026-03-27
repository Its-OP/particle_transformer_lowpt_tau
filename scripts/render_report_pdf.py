import os
from md2pdf.core import md2pdf

reports_directory = "/Users/oleh/Projects/masters/part/reports"
input_path = os.path.join(reports_directory, "biweekly_report_2026_03_26.md")
output_path = os.path.join(reports_directory, "biweekly_report_2026_03_26.pdf")

css = """
@page {
    size: A4;
    margin: 2cm;
}
body {
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #222;
}
h1 {
    font-size: 18pt;
    border-bottom: 2px solid #333;
    padding-bottom: 6px;
}
h2 {
    font-size: 14pt;
    color: #444;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
}
h3 { font-size: 12pt; color: #555; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 10pt;
}
th, td {
    border: 1px solid #ccc;
    padding: 6px 10px;
    text-align: left;
}
th { background-color: #f5f5f5; font-weight: bold; }
code {
    background-color: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10pt;
}
pre {
    background-color: #f4f4f4;
    padding: 12px;
    border-radius: 4px;
    font-size: 9.5pt;
}
img { max-width: 100%; }
"""

md2pdf(
    output_path,
    md_file_path=input_path,
    css_file_path=None,
    base_url=reports_directory,
    raw_css=css,
)

print(f"Saved to {output_path}")
