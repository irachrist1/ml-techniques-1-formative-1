/** Build an editable Word report from the same Markdown and figures as the PDF. */
const fs=require('fs'),path=require('path');
const {Document,Packer,Paragraph,TextRun,Table,TableRow,TableCell,ImageRun,Footer,HeadingLevel,WidthType,ShadingType,ExternalHyperlink,PageNumber,AlignmentType}=require('docx');
const root=__dirname,source=path.join(root,'output/report.md'),width=10226;
function runs(s,opt={}){
 const result=[];const re=/\*\*(.*?)\*\*|`([^`]+)`|\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;let i=0,m;
 while((m=re.exec(s))){if(m.index>i)result.push(new TextRun({text:s.slice(i,m.index),...opt}));if(m[1]!==undefined)result.push(new TextRun({text:m[1],...opt,bold:true}));else if(m[2]!==undefined)result.push(new TextRun({text:m[2],...opt,font:'Courier New'}));else result.push(new ExternalHyperlink({link:m[4],children:[new TextRun({text:m[3],...opt,color:'14645A',underline:{}})]}));i=re.lastIndex;}
 if(i<s.length)result.push(new TextRun({text:s.slice(i),...opt}));return result;
}
const lines=fs.readFileSync(source,'utf8').split(/\r?\n/),children=[];let pageBreak=false;
for(let i=0;i<lines.length;i++){
 let l=lines[i].trim();if(!l)continue;if(l==='---PAGE---'){pageBreak=true;continue;}
 if(l.startsWith('|')){
  const rows=[];while(i<lines.length&&lines[i].trim().startsWith('|')){const c=lines[i].trim().replace(/^\||\|$/g,'').split('|').map(s=>s.trim());if(!c.every(x=>/^[:\- ]+$/.test(x)))rows.push(c);i++;}i--;
  let weights=rows[0].map(()=>1/rows[0].length);if(rows[0][0]==='Selected model')weights=[.17,.16,.67];else if(rows[0][0]==='Model (seed 42)')weights=[.4,.2,.2,.2];
  const widths=weights.map(v=>Math.floor(v*width));widths[widths.length-1]+=width-widths.reduce((a,b)=>a+b,0);
  children.push(new Table({width:{size:width,type:WidthType.DXA},columnWidths:widths,rows:rows.map((row,r)=>new TableRow({tableHeader:r===0,cantSplit:true,children:row.map((cell,c)=>new TableCell({width:{size:widths[c],type:WidthType.DXA},margins:{top:55,bottom:55,left:80,right:80},shading:{type:ShadingType.CLEAR,fill:r===0?'E5EEEB':r%2?'FFFFFF':'F5F7F6'},children:[new Paragraph({spacing:{before:0,after:0,line:200},children:runs(cell,{size:16,bold:r===0})})]}))}))}));
  children.push(new Paragraph({spacing:{after:50},children:[]}));continue;
 }
 const im=l.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
 if(im){const p=path.resolve(path.dirname(source),im[2]),data=fs.readFileSync(p),w=data.readUInt32BE(16),h=data.readUInt32BE(20),scale=Math.min((width/15)/w,450/h);
  children.push(new Paragraph({pageBreakBefore:pageBreak,keepNext:true,spacing:{after:40},children:[new ImageRun({type:'png',data,transformation:{width:Math.round(w*scale),height:Math.round(h*scale)},altText:{title:im[1],description:im[1],name:path.basename(p)}})]}));pageBreak=false;
  children.push(new Paragraph({spacing:{after:120},children:runs(im[1],{size:16,color:'53605D'})}));continue;
 }
 let heading; if(l.startsWith('# ')){heading=HeadingLevel.TITLE;l=l.slice(2);}else if(l.startsWith('## ')){heading=HeadingLevel.HEADING_1;l=l.slice(3);}else if(l.startsWith('### ')){heading=HeadingLevel.HEADING_2;l=l.slice(4);}
 children.push(new Paragraph({heading,pageBreakBefore:pageBreak,keepNext:!!heading,spacing:heading?undefined:{after:120,line:250},children:runs(l,/^\[\d+\]/.test(l)?{size:17}:{})}));pageBreak=false;
}
const doc=new Document({creator:'Christian Tonny',title:'One-step mobile-network traffic forecasting in Milan',styles:{default:{document:{run:{font:'Arial',size:20,color:'24312F'},paragraph:{spacing:{after:120,line:250}}}},paragraphStyles:[{id:'Title',name:'Title',basedOn:'Normal',run:{font:'Arial',size:38,bold:true,color:'000000'},paragraph:{spacing:{before:0,after:220}}},{id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:28,bold:true},paragraph:{outlineLevel:0,spacing:{before:160,after:120}}},{id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:23,bold:true},paragraph:{outlineLevel:1,spacing:{before:120,after:90}}}]},sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:720,bottom:840,left:840,right:840}}},footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.RIGHT,children:[new TextRun({text:'ML Techniques I | Formative Assignment 1 | ',size:14,color:'53605D'}),new TextRun({children:[PageNumber.CURRENT],size:14})]})]})},children}]});
Packer.toBuffer(doc).then(b=>{const target=path.join(root,'output/formative1_report.docx');fs.writeFileSync(target,b);console.log(target);});
