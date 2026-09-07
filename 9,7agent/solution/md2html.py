# -*- coding: utf-8 -*-
"""把 论文.md 转成自包含 HTML(图片 base64 内嵌, 可离线预览)"""
import re, base64, html, os

md = open("论文.md", encoding="utf-8").read()
figs = {}

def fig_uri(name):
    if name in figs:
        return figs[name]
    if not os.path.exists(name):
        return None
    with open(name, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    uri = f"data:image/png;base64,{b64}"
    figs[name] = uri
    return uri

# 1) 将 "> 图 N ...（figs/xx.png）..." 的图注行转为图片块（可含多个图）
def img_blocks(m):
    txt = m.group(0)
    names = re.findall(r"figs/[\w.-]+\.png", txt)
    cap = re.sub(r"^>\s*图\s*\d+\s*", "图 ", txt)
    cap = re.sub(r"[（(]?figs/[\w.-]+\.png[）)]?", "", cap).strip()
    out = ""
    for nm in names:
        uri = fig_uri(nm)
        if uri:
            out += f'<figure style="margin:14px 0;text-align:center"><img src="{uri}" style="max-width:92%;border:1px solid #ddd;border-radius:6px"/><figcaption style="color:#555;font-size:13px;margin-top:6px">{html.escape(cap)}</figcaption></figure>\n'
    return out

md = re.sub(r">[^\n]*figs/[\w.-]+\.png[^\n]*", img_blocks, md)

# 2) 行内公式/块级公式 -> 简单斜体灰字
def inline_math(m):
    return f"<i>{html.escape(m.group(1))}</i>"
md = re.sub(r"\$([^$\n]+?)\$", inline_math, md)
def block_math(m):
    return f'<div style="text-align:center;font-style:italic;color:#333;margin:8px 0;overflow-x:auto">{html.escape(m.group(1))}</div>'
md = re.sub(r"\$\$([^$]+?)\$\$", block_math, md)

lines = md.split("\n")
out = []
in_table = False
in_code = False
table_buf = []

def flush_table(buf):
    rows = []
    for i, ln in enumerate(buf):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        tag = "th" if i == 0 else "td"
        rows.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
    return '<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:10px 0;font-size:13.5px">' + \
           "".join(rows) + "</table></div>"

for ln in lines:
    if ln.strip().startswith("```"):
        in_code = not in_code
        if in_code:
            out.append('<pre style="background:#f6f8fa;padding:10px;border-radius:6px;overflow-x:auto"><code>')
        else:
            out.append("</code></pre>")
        continue
    if in_code:
        out.append(html.escape(ln))
        continue
    if in_table:
        if ln.strip().startswith("|"):
            table_buf.append(ln)
            continue
        out.append(flush_table(table_buf)); table_buf = []; in_table = False
    s = ln.strip()
    if s.startswith("|") and s.endswith("|"):
        in_table = True; table_buf = [ln]
        continue
    if s.startswith("---"):
        out.append('<hr style="border:none;border-top:1px solid #ccc;margin:14px 0"/>')
        continue
    if s.startswith("#### "):
        out.append(f"<h4 style='margin:18px 0 6px'>{html.escape(s[5:])}</h4>")
    elif s.startswith("### "):
        out.append(f"<h3 style='margin:20px 0 6px;border-bottom:2px solid #4C72B0;padding-bottom:4px'>{html.escape(s[4:])}</h3>")
    elif s.startswith("## "):
        out.append(f"<h2 style='margin:26px 0 8px;color:#2b3a67'>{html.escape(s[3:])}</h2>")
    elif s.startswith("# "):
        out.append(f"<h1 style='margin:6px 0 10px;color:#1f2d50'>{html.escape(s[2:])}</h1>")
    elif s.startswith("<figure"):
        out.append(s)
    elif s.startswith("- "):
        out.append(f"<li>{html.escape(s[2:])}</li>")
    elif s.startswith("**表"):
        out.append(f"<p style='font-weight:700;margin:12px 0 4px'>{html.escape(s)}</p>")
    elif re.match(r"^\d+\.\s", s):
        out.append(f"<p style='margin:2px 0'>{html.escape(s)}</p>")
    elif s.startswith("> "):
        out.append(f'<blockquote style="margin:8px 0;padding:2px 12px;border-left:4px solid #4C72B0;color:#444">{html.escape(s[2:])}</blockquote>')
    elif s == "":
        pass
    else:
        out.append(f"<p style='margin:6px 0;line-height:1.7'>{html.escape(s)}</p>")

body = "\n".join(out)
page = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NIPT 时点选择与胎儿异常判定 — 建模论文</title>
<style>body{{font-family:'Noto Sans CJK SC','PingFang SC','Microsoft YaHei',sans-serif;max-width:980px;margin:0 auto;padding:28px 22px;color:#222;background:#fff}}
h1{{font-size:26px}}h2{{font-size:20px}}h3{{font-size:17px}}table td,table th{{border:1px solid #cfcfcf;padding:5px 9px;text-align:left}}
</style></head><body>
{body}
</body></html>"""
open("论文.html", "w", encoding="utf-8").write(page)
import os
print("论文.html 生成, 大小 %.1f MB" % (os.path.getsize("论文.html") / 1e6))
