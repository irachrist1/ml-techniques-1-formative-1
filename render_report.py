"""Render the editable research report Markdown to a paginated PDF."""
import re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether
ROOT=Path(__file__).resolve().parent


def markup(text):
    text=escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`(.+?)`',r'<font name="Courier">\1</font>',text)
    text=re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#14645a">\1</link>',text)
    return text.replace('–','-').replace('—','-').replace('’',"'").replace('±','+/-')


def render(source=None):
    source=Path(source or ROOT/'output/report.md');out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    target=out/'formative1_report.pdf'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=7,textColor=colors.HexColor('#24312f')))
    styles.add(ParagraphStyle(name='CaptionCustom',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=8,textColor=colors.HexColor('#53605d')))
    styles.add(ParagraphStyle(name='ReferenceCustom',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=5,textColor=colors.HexColor('#24312f')))
    styles.add(ParagraphStyle(name='CellCustom',fontName='Helvetica',fontSize=7.7,leading=10))
    styles['Title'].fontSize=21;styles['Title'].leading=25;styles['Title'].textColor=colors.HexColor('#14645a')
    styles['Heading1'].fontSize=14;styles['Heading1'].leading=17
    styles['Heading2'].fontSize=11;styles['Heading2'].leading=14
    story=[];lines=source.read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line=='---PAGE---':story.append(PageBreak());continue
        if line.startswith('|'):
            table_lines=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip());i+=1
            rows=[]
            for row in table_lines:
                cells=[c.strip() for c in row.strip('|').split('|')]
                if all(re.fullmatch(r'[:\- ]+',c) for c in cells):continue
                rows.append([Paragraph(markup(c),styles['CellCustom']) for c in cells])
            widths=[(A4[0]-84)/len(rows[0])]*len(rows[0])
            if 'Selected model' in table_lines[0]: widths=[(A4[0]-84)*v for v in [.17,.16,.67]]
            elif 'Model (seed 42)' in table_lines[0]: widths=[(A4[0]-84)*v for v in [.4,.2,.2,.2]]
            elif 'Experiment' in table_lines[0]: widths=[(A4[0]-84)*v for v in [.28,.28,.22,.22]]
            t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eeeb')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),3.5),('TOPPADDING',(0,0),(-1,-1),3.5),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#60756e')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f6')])]))
            if 'RMSE mean (3 seeds)' in table_lines[0]: t.setStyle(TableStyle([('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
            story.extend([t,Spacer(1,9)]);continue
        image=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
        if image:
            p=source.parent/image.group(2);im=Image(str(p));scale=min((A4[0]-84)/im.imageWidth,330/im.imageHeight)
            im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
            story.append(KeepTogether([im,Paragraph(markup(image.group(1)),styles['CaptionCustom'])]));continue
        if line.startswith('# '):style=styles['Title'];line=line[2:]
        elif line.startswith('## '):style=styles['Heading1'];line=line[3:]
        elif line.startswith('### '):style=styles['Heading2'];line=line[4:]
        elif re.match(r'^\[\d+\]',line):style=styles['ReferenceCustom']
        else:style=styles['BodyCustom']
        story.append(Paragraph(markup(line),style))
    def page(canvas,doc):
        canvas.setFont('Helvetica',7);canvas.setFillColor(colors.HexColor('#53605d'))
        canvas.drawString(42,24,'ML Techniques I | Formative Assignment 1')
        canvas.drawRightString(A4[0]-42,24,str(doc.page))
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=42,title='Milan mobile-network traffic forecasting',author='Christian Tonny')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(target)


if __name__=='__main__':render()
