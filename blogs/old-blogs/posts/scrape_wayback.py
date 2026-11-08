#!/usr/bin/env python3
"""Recover old Blogger posts from the Wayback Machine and convert to markdown."""
import json, re, html, time, sys, urllib.request, urllib.error

CDX = "/tmp/opencode/wayback_cdx.json"
ROWS = json.load(open(CDX))
rows = ROWS[1:]

POST_RE = re.compile(r"tech\.palashkantikundu\.in/20\d\d/\d\d/[a-z0-9\-_]+\.html", re.I)
posts = {}
for ts, url, status in rows:
    m = POST_RE.search(url)
    if not m or status != "200":
        continue
    slug = m.group(0).rstrip("/")
    path = slug.strip("/").replace(".html", "").split("/")
    if len(path) > 1 and path[1].startswith("20"):  # e.g. 2014/12/title
        key = slug
        if key not in posts:
            posts[key] = [ts, url]

def fetch(url, retries=6):
    req = urllib.request.Request(url, headers={"User-Agent": "blog-recovery/1.0"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait = 25 + 15 * i
                print(f"  .. rate-limited/light error {e.code}, sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if i == retries - 1:
                print(f"  !! failed {url}: {e}", file=sys.stderr)
                return None
            time.sleep(3)
        except Exception as e:
            if i == retries - 1:
                print(f"  !! failed {url}: {e}", file=sys.stderr)
                return None
            time.sleep(5)

XPATHS = {
    "title": re.compile(r"<h3 class='post-title entry-title'[^>]*>\s*(?:<a[^>]*>)?([\s\S]*?)(?:</a>)?</h3>"),
    "body": re.compile(r"<div class='.post-body entry-content.'[^>]*>([\s\S]*?)<div class='post-footer"),
    "date": re.compile(r"<abbr class='published'[^>]*title='([^']+)'"),
    "dated2": re.compile(r"<h2 class='date-header'>[^<]*<span>([^<]+)</span>"),
}

def inline_md(s):
    s = re.sub(r"<(b|strong)[^>]*>([\s\S]*?)</\1>", lambda m: "**" + m.group(2).strip() + "**", s)
    s = re.sub(r"<(i|em)[^>]*>([\s\S]*?)</\1>", lambda m: partial_em(m), s)
    s = re.sub(r"<u[^>]*>([\s\S]*?)</u>", r"_\1_", s, flags=re.I)
    s = re.sub(r'<a[^>]*href=["\'](\S+?)["\'][^>]*>([\s\S]*?)</a>',
               lambda m: "[" + clean_txt(m.group(2)) + "](" + clean_href(m.group(1)) + ")" if m.group(2).strip() else f"<{clean_href(m.group(1))}>", s, flags=re.I)
    s = re.sub(r"<img[^>]*src=['\"](\S+?)['\"][^>]*>", lambda m: "![image](" + clean_href(m.group(1)) + ")", s, flags=re.I)
    return s

def partial_em(m):
    txt = clean_txt(m.group(2))
    return "*" + txt + "*" if txt.strip() else ""

def clean_href(h):
    h = html.unescape(h)
    # keep wayback-rewritten links pointing at the live original
    h = re.sub(r"^https?://web\.archive\.org/web/\d+(?:id_)?/", "", h)
    return h

def clean_txt(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()

def block_md(s):
    out = []
    for li in re.findall(r"<li[^>]*>([\s\S]*?)</li>", s, flags=re.I):
        out.append("- " + inline_md(li).strip())
    return "\n".join(out)

def convert(body):
    # drop wayback-inserted blocklists & scripts
    body = re.sub(r"<!--[\s\S]*?-->", "", body)
    body = re.sub(r"<(script|style)[^>]*>[\s\S]*?</\1>", "", body, flags=re.I)
    body = re.sub(r"<a[^>]*name=['\"]more['\"][^>]*>\s*</a>", "", body)
    body = re.sub(r'<div dir="ltr"[^>]*>|</div>\s*(?=<)', "\n", body)

    parts = []
    # ordered/unordered lists
    def list_sub(m):
        items = block_md(m.group(2))
        return "\n" + m.group(1).lower().find("<ul") == -1 and items or items
    # headings before generic inline
    body = re.sub(r"<h([1-6])[^>]*>([\s\S]*?)</h\1>", lambda m: f"\n\n{'#' * int(m.group(1))} " + clean_txt(m.group(2)) + "\n\n", body, flags=re.I)
    body = re.sub(r"<(ul|li|ol)[^>]*>", "\n", body, flags=re.I)
    body = re.sub(r"</(ul|ol|li|table|tr)>", "\n", body, flags=re.I)
    body = re.sub(r"<(td|th|caption)[^>]*>", " | ", body, flags=re.I)
    body = re.sub(r"</(td|th)>", " ", body, flags=re.I)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    body = re.sub(r"<p[^>]*>|</p>", "\n", body, flags=re.I)
    body = re.sub(r"<(div|span|td|tr|table)[^>]*>", "", body, flags=re.I)
    body = re.sub(r"</(div|span|td|tr|table|b|i|u|em|strong|sup|sub|small|center|blockquote)>", "", body, flags=re.I)
    body = re.sub(r"<center[^>]*>|</center>", "", body, flags=re.I)
    body = re.sub(r"<blockquote[^>]*>", "\n> ", body, flags=re.I)
    # escape stray < not part of tags
    body = re.sub(r"<(?![a-zA-Z/])", "&lt;", body)
    # starts of ordered items numbered by writer manually: leave as-is
    t = inline_md(body)
    # collapse whitespace
    lines = [l.strip() for l in t.splitlines()]
    out, blank = [], False
    for l in lines:
        l = re.sub(r"\s+ ?$", "", l)
        if not l:
            if out and not blank:
                out.append("")
            blank = True
        else:
            out.append(l); blank = False
    txt = "\n".join(out)
    txt = html.unescape(txt)
    txt = re.sub(r"\u00a0", " ", txt)
    txt = re.sub(r"^\*\*", "**", txt)
    txt = txt.replace("\n\n\n", "\n\n")
    return txt

def main():
    import os
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/recovered-posts"
    os.makedirs(d, exist_ok=True)
    ok = 0
    from pathlib import Path
    d = Path(d)
    for slug, (ts, url) in sorted(posts.items()):
        path = slug.strip("/").split("/")
        year, month, rawslug = path[1], path[2], path[3]
        url = "https://tech.palashkantikundu.in/" + year + "/" + month + "/" + rawslug
        snap = f"https://web.archive.org/web/{ts}id_/{url}"
        page = fetch(snap)
        time.sleep(8)
        if not page:
            continue
        tm = XPATHS["title"].search(page)
        bstart = re.search(r"<div class=[\"']post-body entry-content[\"']", page)
        bm = None
        if bstart:
            ends = [m.start() for end in ("post-footer", "blog-pager", "post-body entry-content-outter-wrapper", "<h3 class='post-title", "comments")
                    for m in re.finditer(re.escape(end), page[bstart.end():])
                    if m.start() > 0]
            gt = page.find('>', bstart.end())
            ends = [e for e in ends if gt < bstart.end() + e] or ends
            bend = bstart.end() + min(ends) if ends else None
            pos = gt + 1
            bm = type("M", (), {"group": staticmethod(lambda i: page[pos:bend])})() if bend else None
        if not bm:
            print(f"  !! no body for {slug}", file=sys.stderr)
            continue
        title = clean_txt(tm.group(1)) if tm else rawslug
        body = convert(bm.group(1))
        date = ""
        dm = XPATHS["date"].search(page) or XPATHS["dated2"].search(page)
        if dm:
            date = clean_txt(dm.group(1))
        fn = f"{year}-{month}-{rawslug[:-5].replace('_', '-')}.md"
        head = f"---\ntitle: \"{title}\"\ndate: \"{date if date else year + '-' + month}\"\nrecovered_from: \"web.archive.org (snapshot {ts})\"\n---\n\n"
        with open(os.path.join(d, fn), "w") as f:
            f.write(head + body + "\n")
        ok += 1
        print(f"OK {title[:70]} -> {fn}")
    print(f"done: {ok}/{len(posts)} recovered")

main()
