#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the shared and data-driven parts of the website.

Every page keeps its own hand-written content. The blocks between the
`<!-- start NAME -->` and `<!-- end NAME -->` comments are owned by this script:

    head              title, description, canonical, social cards, stylesheets,
                      scripts and (home page) structured data
    header            site header with the menu (home, publications, talks)
    footer            footer links and analytics (home, publications, talks)
    publication list  the paper list, from publications_data.py

Usage:
    python3 build.py           rewrite the pages
    python3 build.py --check   exit with status 1 if a page is out of date
"""

import argparse
import hashlib
import html
import json
import os
import re
import sys

from publications_data import authors, conferences, publications  # noqa: F401

ROOT = os.path.dirname(os.path.abspath(__file__))

SITE = "https://paschalidoud.github.io"
NAME = "Despoina Paschalidou"
ANALYTICS_ID = "G-ZJHZX3DK7N"

# Profiles shown in the footer and listed in the structured data.
SCHOLAR = "https://scholar.google.com/citations?user=zxFlR6sAAAAJ"
GITHUB = "https://github.com/paschalidoud"
TWITTER = "https://twitter.com/paschalidoud_1"
ORCID = ""  # e.g. "https://orcid.org/0000-0000-0000-0000"; hidden while empty
EMAIL = "dpaschalidou@nvidia.com"

HOME_DESCRIPTION = (
    "Despoina Paschalidou is a Senior Research Scientist at the NVIDIA Spatial "
    "Intelligence Lab, working on computer vision and graphics: 3D "
    "reconstruction, generative models, and embodied intelligence."
)

PAGES = [
    {
        "file": "index.html",
        "kind": "site",
        "css": "style.css",
        "title": NAME,
        "description": HOME_DESCRIPTION,
        "url": SITE + "/",
        "image": "figures/me-small.jpg",
        "person": True,
        "papers": "selected",
    },
    {
        "file": "publications.html",
        "kind": "site",
        "css": "style.css",
        "title": "Publications | " + NAME,
        "description": (
            "Publications by Despoina Paschalidou on 3D reconstruction, "
            "generative models for scenes and video, and world models for "
            "embodied intelligence."
        ),
        "url": SITE + "/publications",
        "image": "figures/me-small.jpg",
        "papers": "all",
    },
    {
        "file": "talks.html",
        "kind": "site",
        "css": "style.css",
        "title": "Presentations | " + NAME,
        "description": (
            "Talks and presentations by Despoina Paschalidou on 3D shape and "
            "scene representations, primitive-based reconstruction and "
            "generative models."
        ),
        "url": SITE + "/talks",
        "image": "figures/me-small.jpg",
    },
    {
        "file": "atiss.html",
        "kind": "project",
        "css": "style_project_page.css",
        "title": "ATISS: Autoregressive Transformers for Indoor Scene Synthesis",
        "description": (
            "ATISS is an autoregressive transformer that generates indoor "
            "scenes as unordered sets of objects, enabling layout synthesis, "
            "scene completion and object suggestion. NeurIPS 2021."
        ),
        "url": SITE + "/atiss",
        "image": "projects/atiss/teaser.png",
    },
    {
        "file": "neural_parts.html",
        "kind": "project",
        "css": "style_project_page.css",
        "title": (
            "Neural Parts: Learning Expressive 3D Shape Abstractions with "
            "Invertible Neural Networks"
        ),
        "description": (
            "Neural Parts is a 3D primitive representation that can capture "
            "arbitrarily complex genus-zero shapes, giving geometrically "
            "accurate and semantically meaningful shape abstractions. "
            "CVPR 2021."
        ),
        "url": SITE + "/neural_parts",
        "image": "projects/neural_parts/teaser.png",
    },
]

MENU = [
    ("index.html", "Home", SITE),
    ("publications.html", "Publications", SITE + "/publications"),
    ("talks.html", "Presentations", SITE + "/talks"),
]


def esc(text):
    return html.escape(text, quote=True)


def asset(path):
    """Asset path with a content hash, so browsers never serve a stale copy."""
    with open(os.path.join(ROOT, path), "rb") as f:
        digest = hashlib.sha1(f.read()).hexdigest()[:10]
    return "{}?v={}".format(path, digest)


def indent(text, spaces):
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in text.split("\n"))


# --------------------------------------------------------------------------
# head / header / footer
# --------------------------------------------------------------------------

def person_json_ld():
    same_as = [u for u in (SCHOLAR, GITHUB, TWITTER, ORCID) if u]
    data = {
        "@context": "https://schema.org",
        "@type": "Person",
        "name": NAME,
        "url": SITE + "/",
        "image": SITE + "/figures/me-small.jpg",
        "jobTitle": "Senior Research Scientist",
        "worksFor": {"@type": "Organization", "name": "NVIDIA"},
        "alumniOf": [
            {"@type": "Organization",
             "name": "Max Planck ETH Center for Learning Systems"},
            {"@type": "CollegeOrUniversity", "name": "Stanford University"},
            {"@type": "CollegeOrUniversity",
             "name": "Aristotle University of Thessaloniki"},
        ],
        "sameAs": same_as,
    }
    return json.dumps(data, indent=2, ensure_ascii=False).replace("</", "<\\/")


def build_head(page):
    image = "{}/{}".format(SITE, page["image"])
    card = "summary_large_image" if page["kind"] == "project" else "summary"
    og_type = "website" if page["kind"] == "site" else "article"
    lines = [
        "<title>{}</title>".format(esc(page["title"])),
        '<meta name="description" content="{}">'.format(esc(page["description"])),
        '<meta name="color-scheme" content="light dark">',
        '<link rel="canonical" href="{}">'.format(page["url"]),
        "<!-- OpenGraph (Facebook / LinkedIn / Slack / iMessage) -->",
        '<meta property="og:type" content="{}">'.format(og_type),
        '<meta property="og:url" content="{}">'.format(page["url"]),
        '<meta property="og:title" content="{}">'.format(esc(page["title"])),
        '<meta property="og:description" content="{}">'.format(esc(page["description"])),
        '<meta property="og:image" content="{}">'.format(image),
        "<!-- Twitter / X card -->",
        '<meta name="twitter:card" content="{}">'.format(card),
        '<meta name="twitter:title" content="{}">'.format(esc(page["title"])),
        '<meta name="twitter:description" content="{}">'.format(esc(page["description"])),
        '<meta name="twitter:image" content="{}">'.format(image),
        '<link rel="stylesheet" type="text/css" href="{}">'.format(asset("common.css")),
        '<link rel="stylesheet" type="text/css" href="{}">'.format(asset(page["css"])),
        "<script>document.documentElement.classList.add('js');</script>",
        '<script defer src="{}"></script>'.format(asset("site.js")),
    ]
    if page.get("person"):
        lines += [
            '<script type="application/ld+json">',
            person_json_ld(),
            "</script>",
        ]
    return "\n" + indent("\n".join(lines), 8) + "\n        "


def build_header(page):
    items = []
    for filename, label, url in MENU:
        current = ' aria-current="page"' if filename == page["file"] else ""
        items.append(
            '<li>\n    <a href="{}"{}>{}</a>\n</li>'.format(url, current, label)
        )
    return "\n" + indent(
        '<div class="header">\n'
        '    <div class="section">\n'
        '        <div class="myname"><a href="{site}/">{name}</a></div>\n'
        '        <nav class="menu" aria-label="Main">\n'
        '            <ul>\n{items}\n            </ul>\n'
        '        </nav>\n'
        '    </div>\n'
        '</div>'.format(site=SITE, name=NAME, items=indent("\n".join(items), 16)),
        8,
    ) + "\n        "


def build_footer(page):
    links = [("Google Scholar", SCHOLAR), ("GitHub", GITHUB)]
    if ORCID:
        links.append(("ORCID", ORCID))
    links.append(("Email", "mailto:" + EMAIL))
    anchors = "\n".join(
        '        <a href="{}" rel="me">{}</a>'.format(esc(url), label)
        for label, url in links
    )
    text = (
        '<footer class="site-footer">\n'
        '    <div class="section">\n'
        '{anchors}\n'
        '    </div>\n'
        '</footer>\n'
        '\n'
        '<!-- Google tag (gtag.js) - Google Analytics 4 -->\n'
        '<script async src="https://www.googletagmanager.com/gtag/js?id={gid}"></script>\n'
        '<script>\n'
        '    window.dataLayer = window.dataLayer || [];\n'
        '    function gtag(){{dataLayer.push(arguments);}}\n'
        "    gtag('js', new Date());\n"
        "    gtag('config', '{gid}');\n"
        '</script>'
    ).format(anchors=anchors, gid=ANALYTICS_ID)
    return "\n" + indent(text, 8) + "\n        "


# --------------------------------------------------------------------------
# publication list
# --------------------------------------------------------------------------

def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def paper_id(paper):
    """Stable anchor for a paper: its BibTeX key, else a slug of the title."""
    for link in paper.links:
        if link.name == "Bibtex" and link.text:
            m = re.search(r"@\w+\s*\{\s*([^,\s}]+)", link.text)
            if m:
                return slug(m.group(1))
    return slug(paper.title)


def render_paper(p, pid):
    def author(a):
        if a.is_me:
            return '<strong class="author">{}</strong>'.format(esc(a.name))
        return '<a href="{}" class="author">{}</a>'.format(esc(a.url), esc(a.name))

    conference = "{}, {}".format(esc(p.conference.name), p.year)
    if p.special is not None:
        conference += '<div class="special">   ({})</div>'.format(esc(p.special))

    buttons, panels = [], []
    for link in p.links:
        if link.url is not None:
            buttons.append('<a href="{}" data-type="{}">{}</a>'.format(
                esc(link.url), esc(link.name), esc(link.name)))
            continue
        panel_id = "{}-{}".format(pid, slug(link.name))
        buttons.append(
            '<a href="#{id}" data-type="{n}" role="button" aria-expanded="false" '
            'aria-controls="{id}">{n}</a>'.format(id=panel_id, n=esc(link.name)))
        body = link.html if link.html is not None else (
            "<pre>" + html.escape(link.text, quote=False) + "</pre>")
        panels.append(
            '<div class="link-content" id="{}" role="region" aria-label="{}">{}</div>'
            .format(panel_id, esc(link.name), body))

    image = ('<img src="{}" alt="{}" />'.format(esc(p.image), esc(p.title))
             if p.image is not None else "&nbsp;")
    return (
        '<div class="row paper" id="{pid}">'
        '<div class="image">{image}</div>'
        '<div class="content">'
        '<div class="paper-title"><a href="{url}">{title}</a></div>'
        '<div class="conference">{conference}</div>'
        '<div class="authors">{authors}</div>'
        '<div class="links">{buttons} {panels}</div>'
        '</div></div>'
    ).format(
        pid=pid, image=image, url=esc(p.url), title=esc(p.title),
        conference=conference,
        authors=", ".join(author(a) for a in p.authors),
        buttons=" ".join(buttons), panels="".join(panels),
    )


def build_publication_list(which):
    papers = [p for p in publications if which == "all" or p.selected]
    ids = [paper_id(p) for p in papers]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit("Duplicate paper ids: " + ", ".join(dupes))
    return "\n" + "\n".join(render_paper(p, i) for p, i in zip(papers, ids)) + "\n"


# --------------------------------------------------------------------------

def replace_block(text, name, content, filename):
    start_tag = "<!-- start {} -->".format(name)
    end_tag = "<!-- end {} -->".format(name)
    start = text.find(start_tag)
    end = text.find(end_tag, start)
    if start < 0 or end < 0:
        raise SystemExit("{}: missing '{}' fence comments".format(filename, name))
    return text[:start + len(start_tag)] + content + text[end:]


def build_page(page):
    path = os.path.join(ROOT, page["file"])
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new = replace_block(text, "head", build_head(page), page["file"])
    if page["kind"] == "site":
        new = replace_block(new, "header", build_header(page), page["file"])
        new = replace_block(new, "footer", build_footer(page), page["file"])
    if page.get("papers"):
        new = replace_block(new, "publication list",
                            build_publication_list(page["papers"]), page["file"])
    return path, text, new


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if any page is out of date")
    args = parser.parse_args()

    stale = []
    for page in PAGES:
        path, old, new = build_page(page)
        if new != old:
            stale.append(page["file"])
            if not args.check:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
    if args.check:
        if stale:
            print("Out of date: " + ", ".join(stale), file=sys.stderr)
            sys.exit(1)
        print("All pages up to date.")
    else:
        print("Updated: " + (", ".join(stale) if stale else "nothing to do"))


if __name__ == "__main__":
    main()
