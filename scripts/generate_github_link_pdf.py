from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "github_link"
PDF = OUT / "github_project_link.pdf"
TXT = OUT / "github_project_link.txt"
GITHUB_URL = "https://github.com/hemu77/UA_Capstone"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TXT.write_text(GITHUB_URL + "\n", encoding="utf-8")

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=letter,
        rightMargin=0.9 * inch,
        leftMargin=0.9 * inch,
        topMargin=1.2 * inch,
        bottomMargin=1.2 * inch,
        title="GitHub Project Link",
    )

    title_style = ParagraphStyle(
        "Title",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        alignment=1,
        textColor=colors.HexColor("#111111"),
        spaceAfter=24,
    )
    link_style = ParagraphStyle(
        "Link",
        fontName="Helvetica",
        fontSize=15,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#0645AD"),
    )

    story = [
        Spacer(1, 2.0 * inch),
        Paragraph("GitHub Project Link", title_style),
        Paragraph(f'<a href="{GITHUB_URL}">{GITHUB_URL}</a>', link_style),
    ]
    doc.build(story)
    print(PDF)
    print(TXT)


if __name__ == "__main__":
    main()
