#!/usr/bin/env python3
"""Post-process recovered posts: heal unclosed <b>/<i>, empty image alts."""
import re, glob

for fn in glob.glob('/tmp/opencode/recovered-posts/*.md'):
    body = open(fn).read().split('\n---\n', 1)[-1]
    text = body

    # empty alt link -> image
    text = re.sub(r'\[\]\((\S+?)\)', r'![image](\1)', text)

    # unclosed/paired italics & bold -> markdown markers, preserving text
    text = re.sub(r'<(i|em)>([^<]*)', lambda m: ('*' + m.group(2)) if m.group(2).strip() else '', text, flags=re.S)
    text = re.sub(r'<(b|strong)>([^<]*)', lambda m: ('**' + m.group(2)) if m.group(2).strip() else '', text, flags=re.S)
    text = re.sub(r'</(i|em|b|strong)>', '', text)

    # stray entity/nbsp leftovers
    text = text.replace('\u00a0', ' ')

    front, _ = open(fn).read().split('\n---\n', 1)
    big = re.split(r'\n---\n', open(fn).read(), maxsplit=1)
    open(fn, 'w').write(big[0] + '\n---\n' + text)
    print('fixed', fn.split('/')[-1])
