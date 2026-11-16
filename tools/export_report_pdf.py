"""Export the technical report with tables and vector illustrations.

Requirements: reportlab>=4, markdown-it-py>=3. Install DejaVu Sans fonts.
Run from any directory: python tools/export_report_pdf.py
"""
from pathlib import Path
from html import escape
import re
from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/Cache_Aware_GNN_Distillation_Technical_Report.md'
OUTPUT=SOURCE.with_suffix('.pdf')
BASE='https://github.com/MeghanaShanthappa/cache-aware-gnn-distillation/blob/main/'
FONT=Path('/usr/share/fonts/truetype/dejavu')
for name,file in [('Body','DejaVuSans.ttf'),('Body-Bold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body',boldItalic='Body-Bold')
ink=colors.HexColor('#172B43');blue=colors.HexColor('#185D83');light=colors.HexColor('#EAF2F7')
styles={
 'body':ParagraphStyle('body',fontName='Body',fontSize=9,leading=13.5,textColor=ink,spaceAfter=8),
 'title':ParagraphStyle('title',fontName='Body-Bold',fontSize=23,leading=29,textColor=ink,spaceAfter=16),
 'h1':ParagraphStyle('h1',fontName='Body-Bold',fontSize=15,leading=20,textColor=blue,spaceBefore=14,spaceAfter=8,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='Body-Bold',fontSize=11.5,leading=16,textColor=ink,spaceBefore=10,spaceAfter=6,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='Body-Bold',fontSize=10,leading=14,textColor=ink,spaceBefore=7,spaceAfter=5,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=7,leading=10,textColor=ink),
 'head':ParagraphStyle('head',fontName='Body-Bold',fontSize=7,leading=10,textColor=colors.white),
 'caption':ParagraphStyle('caption',fontName='Body',fontSize=8,leading=11,textColor=blue,spaceAfter=10,alignment=TA_CENTER),
}

def inline(token):
    out=[]
    for t in token.children or []:
        if t.type=='text':out.append(escape(t.content))
        elif t.type=='strong_open':out.append('<b>')
        elif t.type=='strong_close':out.append('</b>')
        elif t.type=='em_open':out.append('<i>')
        elif t.type=='em_close':out.append('</i>')
        elif t.type=='code_inline':out.append('<font name="Mono" size="7.5">'+escape(t.content)+'</font>')
        elif t.type=='link_open':
            url=t.attrGet('href')
            if not re.match(r'^https?://',url):url=BASE+url.removeprefix('../')
            out.append('<link href="'+escape(url,quote=True)+'" color="#185D83">')
        elif t.type=='link_close':out.append('</link>')
        elif t.type in ['softbreak','hardbreak']:out.append(' ')
        else:raise ValueError('Unsupported inline token '+t.type)
    return ''.join(out)

def box(d,x,y,w,h,lines):
    d.add(Rect(x,y,w,h,rx=6,fillColor=light,strokeColor=blue,strokeWidth=.8))
    for i,line in enumerate(lines):d.add(String(x+w/2,y+h/2+(len(lines)-1)*6-i*12-3,line,fontName='Body',fontSize=8,textAnchor='middle',fillColor=ink))

def line(d,x,y,xx,yy,arrow=False):
    d.add(Line(x,y,xx,yy,strokeColor=blue,strokeWidth=1))
    if arrow:d.add(Polygon([xx,yy,xx-6,yy+3,xx-6,yy-3],fillColor=blue,strokeColor=blue))

def diagram(product):
    d=Drawing(490,155 if product else 125)
    if product:
        for x,y,label in [(10,108,'Keyboard'),(10,60,'USB hub'),(10,12,'Laptop'),(344,12,'Mouse pad')]:box(d,x,y,115,32,[label])
        box(d,187,60,148,44,['Wireless mouse','Category unknown'])
        for x,y in [(125,124),(125,76),(125,28),(344,28)]:line(d,x,y,187 if x==125 else 335,82)
        line(d,10,124,2,124);line(d,2,124,2,28);line(d,2,28,10,28)
    else:
        box(d,0,48,100,38,['Product query']);box(d,135,48,125,38,['Prediction cached?'])
        box(d,310,83,175,34,['Saved teacher prediction']);box(d,310,5,175,42,['Student prediction','using product features'])
        line(d,100,67,135,67,True);line(d,260,67,287,67);line(d,287,67,287,100);line(d,287,100,310,100,True)
        line(d,287,67,287,26);line(d,287,26,310,26,True)
        d.add(String(275,106,'Yes',fontName='Body',fontSize=8,fillColor=ink));d.add(String(275,12,'No',fontName='Body',fontSize=8,fillColor=ink))
    return d


def main():
    tokens=MarkdownIt('commonmark').enable('table').parse(SOURCE.read_text())
    story=[];i=0;first=True;diagram_count=0;tables=0
    while i<len(tokens):
        t=tokens[i]
        if t.type=='heading_open':
            text=inline(tokens[i+1]);level=int(t.tag[1:]);style='title' if first else ('h1' if level==1 else 'h2' if level==2 else 'h3')
            story.append(Paragraph(text,styles[style]));first=False;i+=3;continue
        if t.type=='paragraph_open':
            story.append(Paragraph(inline(tokens[i+1]),styles['body']));i+=3;continue
        if t.type in ['bullet_list_open','ordered_list_open']:
            ordered=t.type=='ordered_list_open';number=0;i+=1
            while tokens[i].type not in ['bullet_list_close','ordered_list_close']:
                if tokens[i].type=='list_item_open':number+=1
                if tokens[i].type=='inline':story.append(Paragraph((str(number)+'. ' if ordered else '• ')+inline(tokens[i]),styles['body']))
                i+=1
            i+=1;continue
        if t.type=='table_open':
            rows=[];row=[];header=False;i+=1
            while tokens[i].type!='table_close':
                z=tokens[i]
                if z.type=='thead_open':header=True
                elif z.type=='thead_close':header=False
                elif z.type=='tr_open':row=[]
                elif z.type=='inline':row.append(Paragraph(inline(z),styles['head' if header else 'cell']))
                elif z.type=='tr_close':rows.append(row)
                i+=1
            count=len(rows[0]);width=491
            widths=([75,60,65,76,71,73,71] if count==7 else [88,65,92,82,82,82])
            widths=[w*width/sum(widths) for w in widths]
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),blue),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,-1),(-1,-1),.5,blue)]))
            story.extend([KeepTogether([table]),Spacer(1,10)]);tables+=1;i+=1;continue
        if t.type=='fence':
            if t.info=='mermaid':
                diagram_count+=1;story.append(KeepTogether([diagram(diagram_count==1),Paragraph('Illustrative purchase graph' if diagram_count==1 else 'Inference routing: no online teacher execution',styles['caption'])]))
            elif t.info=='math':
                text=('k = floor(0.10 × 2,449,029) = 244,902.' if '\\lfloor' in t.content else 'L<sub>S</sub> = λ CE<sub>w</sub> + (1 − λ) KL<sub>w</sub>(p<sub>T</sub> ∥ p<sub>S</sub>) + β L<sub>energy</sub> + γ L<sub>OSA</sub>')
                story.append(Paragraph(text,ParagraphStyle('formula',parent=styles['body'],backColor=light,borderPadding=9,spaceBefore=10,spaceAfter=14)))
            else:raise ValueError('Unsupported fence '+t.info)
            i+=1;continue
        raise ValueError('Unsupported block token '+t.type)
    assert tables==2 and diagram_count==2
    def footer(c,doc):
        c.saveState();c.setStrokeColor(light);c.line(52,44,A4[0]-52,44)
        c.setFont('Body',7);c.setFillColor(blue);c.drawString(52,30,'Cache-aware GNN distillation | Technical report');c.drawRightString(A4[0]-52,30,str(doc.page));c.restoreState()
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=48,bottomMargin=58,title='Cache-aware GNN Distillation: Architecture, Techniques and Evaluation',author='Meghana Shanthappa')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUTPUT)

if __name__=='__main__':main()
