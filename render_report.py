"""Render the editable research report Markdown to a paginated PDF.

This is a deliberately small Markdown subset: headings, paragraphs, pipe tables,
images, reference lines and an explicit `---PAGE---` break. It is not a general
Markdown renderer, and it raises rather than silently producing a table that
overflows the printable width.
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent

#: Left and right margins are 42pt each, so this is the usable text width.
PRINTABLE_WIDTH = A4[0] - 84

#: Relative column widths for the tables whose default even split reads badly.
#: Keyed by a phrase that appears in the table's header row.
COLUMN_WIDTHS = {
    'Selected model': [.17, .16, .67],
    'Model (seed 42)': [.4, .2, .2, .2],
    'Experiment': [.24, .23, .18, .13, .22],
}


def markup(text: str) -> str:
    """Convert the inline Markdown subset to reportlab's mini-HTML.

    Typographic characters are replaced because the built-in Helvetica and
    Courier fonts have no glyphs for them and would render as black boxes.
    """
    text = escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)
    text = re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#14645a">\1</link>', text)
    return text.replace('–', '-').replace('—', '-').replace('’', "'").replace('±', '+/-')


def build_styles():
    """Paragraph styles for the report body, captions, references and table cells."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom', fontName='Helvetica', fontSize=9.5, leading=13,
                              spaceAfter=7, textColor=colors.HexColor('#24312f')))
    styles.add(ParagraphStyle(name='CaptionCustom', fontName='Helvetica', fontSize=8, leading=10,
                              spaceAfter=8, textColor=colors.HexColor('#53605d')))
    styles.add(ParagraphStyle(name='ReferenceCustom', fontName='Helvetica', fontSize=8.5, leading=11,
                              spaceAfter=5, textColor=colors.HexColor('#24312f')))
    styles.add(ParagraphStyle(name='CellCustom', fontName='Helvetica', fontSize=7.7, leading=10))
    styles['Title'].fontSize = 21
    styles['Title'].leading = 25
    styles['Title'].textColor = colors.HexColor('#14645a')
    styles['Heading1'].fontSize = 14
    styles['Heading1'].leading = 17
    styles['Heading2'].fontSize = 11
    styles['Heading2'].leading = 14
    styles['Heading1'].keepWithNext = True
    styles['Heading2'].keepWithNext = True
    return styles


def build_table(table_lines: list[str], styles) -> Table:
    """Turn consecutive pipe-table lines into a styled reportlab Table.

    Raises:
        ValueError: If a configured width list does not match the column count,
            or the widths do not add up to the printable width. Either would put
            content off the page, which is worse than failing loudly.
    """
    rows = []
    for row in table_lines:
        cells = [c.strip() for c in row.strip('|').split('|')]
        if all(re.fullmatch(r'[:\- ]+', c) for c in cells):
            continue  # the |---|---| alignment row carries no content
        rows.append([Paragraph(markup(c), styles['CellCustom']) for c in cells])

    widths = [PRINTABLE_WIDTH / len(rows[0])] * len(rows[0])
    for phrase, fractions in COLUMN_WIDTHS.items():
        if phrase in table_lines[0]:
            widths = [PRINTABLE_WIDTH * v for v in fractions]
            break
    if len(widths) != len(rows[0]):
        raise ValueError('Table width count must match columns')
    if abs(sum(widths) - PRINTABLE_WIDTH) > 1e-6:
        raise ValueError('Table exceeds printable width')

    table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5eeeb')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#60756e')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7f6')]),
    ]))
    if 'RMSE mean (3 seeds)' in table_lines[0]:
        # This table is the tallest one; tighten it so it stays on a single page.
        table.setStyle(TableStyle([('TOPPADDING', (0, 0), (-1, -1), 2),
                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]))
    return table


def build_story(lines: list[str], source: Path, styles) -> list:
    """Walk the Markdown lines once and emit the reportlab flowable sequence."""
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == '---PAGE---':
            story.append(PageBreak())
            continue
        if line.startswith('|'):
            table_lines = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            story.extend([build_table(table_lines, styles), Spacer(1, 9)])
            continue
        image = re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)', line)
        if image:
            picture = Image(str(source.parent / image.group(2)))
            scale = min(PRINTABLE_WIDTH / picture.imageWidth, 330 / picture.imageHeight)
            picture.drawWidth = picture.imageWidth * scale
            picture.drawHeight = picture.imageHeight * scale
            # Keep the caption with its figure rather than orphaning it overleaf.
            story.append(KeepTogether([picture, Paragraph(markup(image.group(1)), styles['CaptionCustom'])]))
            continue
        if line.startswith('# '):
            style, line = styles['Title'], line[2:]
        elif line.startswith('## '):
            style, line = styles['Heading1'], line[3:]
        elif line.startswith('### '):
            style, line = styles['Heading2'], line[4:]
        elif re.match(r'^\[\d+\]', line):
            style = styles['ReferenceCustom']
        else:
            style = styles['BodyCustom']
        story.append(Paragraph(markup(line), style))
    return story


def render(source: str | Path | None = None) -> Path:
    """Render `source` (default `output/report.md`) to `output/pdf/`."""
    source = Path(source or ROOT / 'output/report.md')
    out = ROOT / 'output/pdf'
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'formative1_report.pdf'
    styles = build_styles()
    story = build_story(source.read_text().splitlines(), source, styles)

    def page(canvas, doc):
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#53605d'))
        canvas.drawString(42, 24, 'ML Techniques I | Formative Assignment 1')
        canvas.drawRightString(A4[0] - 42, 24, str(doc.page))

    doc = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=36, bottomMargin=42,
                            title='Milan mobile-network traffic forecasting', author='Christian Tonny')
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(target)
    return target


if __name__ == '__main__':
    render()
