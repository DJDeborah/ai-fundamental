"""Render the final Chinese research Markdown into a paginated PDF.

The Markdown is the editable source of truth. This builder preserves its
headings, tables, figures, code, hyperlinks, and references for sharing.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, LongTable, Paragraph, SimpleDocTemplate,
    Spacer, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "FINAL_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "0.1B_transformer_final_report.pdf"
PAGE_W, PAGE_H = A4
MARGIN = 47
WIDTH = PAGE_W - 2 * MARGIN


def register_fonts():
    font = Path(r"C:\Windows\Fonts\simhei.ttf")
    if not font.exists():
        raise FileNotFoundError("Chinese font C:/Windows/Fonts/simhei.ttf is required")
    pdfmetrics.registerFont(TTFont("SimHei", str(font)))
    pdfmetrics.registerFont(TTFont("SimHei-Bold", str(font)))


def styles():
    common = dict(fontName="SimHei", wordWrap="CJK", textColor=colors.HexColor("#20252b"))
    return {
        "title": ParagraphStyle("TitleZH", fontSize=18, leading=26, spaceAfter=9,
                                alignment=TA_CENTER, **common),
        "h2": ParagraphStyle("H2ZH", fontSize=12.8, leading=19, spaceBefore=15,
                             spaceAfter=7, keepWithNext=True, **common),
        "h3": ParagraphStyle("H3ZH", fontSize=10.9, leading=16, spaceBefore=9,
                             spaceAfter=5, keepWithNext=True, **common),
        "h4": ParagraphStyle("H4ZH", fontSize=9.5, leading=14, spaceBefore=8,
                             spaceAfter=4, keepWithNext=True, **common),
        "body": ParagraphStyle("BodyZH", fontSize=8.7, leading=13.4, spaceAfter=5,
                               alignment=TA_LEFT, **common),
        "caption": ParagraphStyle("CaptionZH", fontSize=8, leading=12,
                                  textColor=colors.HexColor("#49535d"), spaceAfter=9,
                                  **{k: v for k, v in common.items() if k != "textColor"}),
        "table": ParagraphStyle("TableZH", fontSize=7.2, leading=10.5,
                                spaceAfter=0, **common),
        "code": ParagraphStyle("CodeZH", fontName="SimHei", fontSize=7,
                               leading=9.5, textColor=colors.HexColor("#16252e"),
                               spaceAfter=0),
    }


def inline(node):
    if isinstance(node, NavigableString):
        return html.escape(str(node), quote=False)
    if not isinstance(node, Tag):
        return ""
    child = "".join(inline(c) for c in node.children)
    if node.name in ("strong", "b"):
        return f"<b>{child}</b>"
    if node.name in ("em", "i"):
        return f"<i>{child}</i>"
    if node.name == "code":
        return f'<font color="#174f70">{child}</font>'
    if node.name == "a":
        href = node.get("href", "")
        if href and not re.match(r"^[a-z]+://", href):
            href = (SOURCE.parent / href).resolve().as_uri()
        return f'<link href="{html.escape(href, quote=True)}" color="#176181">{child}</link>'
    if node.name == "br":
        return "<br/>"
    return child


def para(node, style):
    return Paragraph("".join(inline(c) for c in node.children), style)


def build_table(node, st):
    rows = []
    for tr in node.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        rows.append([para(cell, st["table"]) for cell in cells])
    if not rows:
        return None
    ncols = max(map(len, rows))
    for row in rows:
        row.extend([""] * (ncols - len(row)))
    widths = [WIDTH / ncols] * ncols
    table = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
        ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#cbd5dc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return table


def add_list(node, out, st, level=0, ordered=False):
    for i, li in enumerate(node.find_all("li", recursive=False), 1):
        body = "".join(inline(c) for c in li.children if not (isinstance(c, Tag) and c.name in ("ul", "ol")))
        prefix = f"{i}. " if ordered else "- "
        out.append(Paragraph(prefix + body, ParagraphStyle(
            f"List{level}{i}", parent=st["body"], leftIndent=12 + level * 14,
            firstLineIndent=-9, fontSize=8.3, leading=12, spaceAfter=2)))
        for sub in li.find_all(["ul", "ol"], recursive=False):
            add_list(sub, out, st, level + 1, sub.name == "ol")


def page_decor(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#48687c"))
    canvas.setLineWidth(.65)
    canvas.line(MARGIN, PAGE_H - 39, PAGE_W - MARGIN, PAGE_H - 39)
    canvas.setFont("SimHei", 7)
    canvas.setFillColor(colors.HexColor("#526271"))
    canvas.drawString(MARGIN, PAGE_H - 32, "0.1B Transformer · 最终实测报告")
    canvas.drawRightString(PAGE_W - MARGIN, 28, str(doc.page))
    canvas.restoreState()


def main():
    register_fonts()
    st = styles()
    raw = SOURCE.read_text(encoding="utf-8")
    rendered = markdown.markdown(raw, extensions=["tables", "fenced_code"])
    soup = BeautifulSoup(rendered, "html.parser")
    story = []
    for node in soup.children:
        if not isinstance(node, Tag):
            continue
        if node.name == "h1":
            story.append(Spacer(1, 16))
            story.append(para(node, st["title"]))
            story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#476c84")))
            story.append(Spacer(1, 11))
        elif node.name == "h2":
            story.append(para(node, st["h2"]))
        elif node.name == "h3":
            story.append(para(node, st["h3"]))
        elif node.name == "h4":
            story.append(para(node, st["h4"]))
        elif node.name == "p":
            img = node.find("img", recursive=False)
            if img:
                path = (SOURCE.parent / img["src"]).resolve()
                with PILImage.open(path) as im:
                    iw, ih = im.size
                width = min(WIDTH, 485)
                height = width * ih / iw
                story.append(Spacer(1, 6))
                story.append(Image(str(path), width=width, height=height, hAlign="CENTER"))
            elif node.get_text().startswith("图 "):
                story.append(para(node, st["caption"]))
            else:
                story.append(para(node, st["body"]))
        elif node.name == "table":
            tbl = build_table(node, st)
            if tbl:
                story.extend([Spacer(1, 3), tbl, Spacer(1, 8)])
        elif node.name in ("ul", "ol"):
            add_list(node, story, st, ordered=node.name == "ol")
        elif node.name == "pre":
            code = node.get_text().strip().splitlines()
            lines = []
            for line in code:
                while len(line) > 105:
                    lines.append(line[:105])
                    line = "  " + line[105:]
                lines.append(line)
            txt = "<br/>".join(html.escape(line).replace(" ", "&nbsp;") for line in lines)
            story.append(Paragraph(txt, st["code"]))
            story.append(Spacer(1, 5))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=MARGIN,
                            rightMargin=MARGIN, topMargin=47, bottomMargin=31,
                            title="从原始文本到 0.1B Transformer：最终实测报告",
                            author="0.1B Research Notebook")
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT)


if __name__ == "__main__":
    main()
