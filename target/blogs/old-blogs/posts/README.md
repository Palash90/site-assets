# Recovered old Blogger posts

28 markdown posts recovered from the defunct blog at `tech.palashkantikundu.in`
(Blogger account deleted; no XML export exists).

- **Source:** Wayback Machine snapshots (`web.archive.org`), fetched 2026-11-09
- **Original era:** Dec 2014 – Mar 2020 (Java, Python, C, Unix/CLI series)
- **Curated 2026-11-16:** the 2013–15 Java/Big Data era was reduced to two essays
  (`2014-12-programming-is-fun-70.md`, `2014-12-what-is-computer-program.md`) —
  the dated tutorials and config posts were deliberately deleted (recoverable
  from git history if ever needed). A CDX sweep (2026-11-16) proved Wayback
  holds captures for more posts than the original scrape found, including
  `data`, `internet-defined-simply` (restored, then dropped again as it adds
  little beyond the two 2014 essays), `website-or-web-application`
  and ~45 more Java-era posts — see git history and the 2026-11-16 session log.
- **Converter:** `scrape_wayback.py` (kept in this folder for reference) +
  `postfix.py` — Blogger HTML → markdown, Blogger `<b>`-hanging style healed
- **Provenance:** every file's front matter carries the exact snapshot timestamp
- **Images:** still hosted on `*.bp.blogspot.com` and verifiably live; consider
  re-hosting them in this repo to reduce hard dependency on Blogger's CDN
- The 2024 Grafana/AWS post was not recovered here — it was already ported to
  `blogs/aws-grafana/`
