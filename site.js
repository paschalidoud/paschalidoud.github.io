/**
 * Behaviour shared by every page:
 *  - Abstract / BibTeX toggles on the paper lists (keyboard accessible, and
 *    reflected in the URL so a link like publications#wu2026neurips-bibtex
 *    opens that panel)
 *  - a "Copy" button on every BibTeX block
 */
(function () {
    "use strict";

    // ---- Abstract / BibTeX panels ---------------------------------------

    function setOpen(panel, open) {
        panel.classList.toggle("open", open);
        document.querySelectorAll('a[aria-controls="' + panel.id + '"]').forEach(function (a) {
            a.setAttribute("aria-expanded", open ? "true" : "false");
        });
    }

    function closeSiblings(panel) {
        Array.prototype.forEach.call(panel.parentNode.children, function (el) {
            if (el !== panel && el.classList.contains("link-content")) {
                setOpen(el, false);
            }
        });
    }

    function updateHash(id) {
        try {
            history.replaceState(null, "", location.pathname + location.search + (id ? "#" + id : ""));
        } catch (e) { /* e.g. file:// in some browsers; the toggle still works */ }
    }

    function hashPanel() {
        var id = decodeURIComponent(location.hash.slice(1));
        var panel = id && document.getElementById(id);
        return panel && panel.classList.contains("link-content") ? panel : null;
    }

    function openFromHash() {
        var panel = hashPanel();
        if (!panel) {
            return;
        }
        closeSiblings(panel);
        setOpen(panel, true);
        var paper = panel.closest(".paper");
        if (paper) {
            paper.scrollIntoView();
        }
    }

    // Images shift the layout while they load, so scroll to the opened panel
    // once more when everything is in place, unless the reader already moved.
    var moved = false;
    ["wheel", "touchmove", "keydown"].forEach(function (name) {
        window.addEventListener(name, function () { moved = true; }, { once: true, passive: true });
    });
    window.addEventListener("load", function () {
        var panel = hashPanel();
        var paper = panel && panel.closest(".paper");
        if (paper && !moved) {
            paper.scrollIntoView();
        }
    });

    document.addEventListener("click", function (ev) {
        var toggle = ev.target.closest && ev.target.closest("a[aria-controls]");
        if (!toggle) {
            return;
        }
        var panel = document.getElementById(toggle.getAttribute("aria-controls"));
        if (!panel) {
            return;
        }
        ev.preventDefault();
        var open = !panel.classList.contains("open");
        closeSiblings(panel);
        setOpen(panel, open);
        updateHash(open ? panel.id : "");
    });

    // role="button" links should also react to the space bar
    document.addEventListener("keydown", function (ev) {
        if (ev.key === " " && ev.target.matches && ev.target.matches('a[role="button"]')) {
            ev.preventDefault();
            ev.target.click();
        }
    });

    window.addEventListener("hashchange", openFromHash);
    openFromHash();

    // ---- Copy button on BibTeX blocks -----------------------------------

    function legacyCopy(text) {
        var area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.style.cssText = "position:fixed;top:0;left:0;opacity:0";
        document.body.appendChild(area);
        area.select();
        var ok = false;
        try { ok = document.execCommand("copy"); } catch (e) { /* reported below */ }
        document.body.removeChild(area);
        return ok;
    }

    function copyText(text) {
        var viaApi = navigator.clipboard && window.isSecureContext
            ? navigator.clipboard.writeText(text)
            : Promise.reject(new Error("clipboard API unavailable"));
        return viaApi.catch(function () {
            return legacyCopy(text) ? undefined : Promise.reject(new Error("copy failed"));
        });
    }

    document.querySelectorAll(".link-content pre, .bibtex pre").forEach(function (pre) {
        var wrap = document.createElement("div");
        wrap.className = "code-block";
        pre.parentNode.insertBefore(wrap, pre);
        wrap.appendChild(pre);

        var button = document.createElement("button");
        button.type = "button";
        button.className = "copy-btn";
        button.textContent = "Copy";
        button.setAttribute("aria-label", "Copy BibTeX to clipboard");
        wrap.appendChild(button);

        var timer;
        button.addEventListener("click", function () {
            copyText(pre.textContent.trim() + "\n").then(function () {
                button.textContent = "Copied!";
                button.setAttribute("data-copied", "");
            }, function () {
                button.textContent = "Press Ctrl+C";
                var range = document.createRange();
                range.selectNodeContents(pre);
                var sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(range);
            }).then(function () {
                clearTimeout(timer);
                timer = setTimeout(function () {
                    button.textContent = "Copy";
                    button.removeAttribute("data-copied");
                }, 1800);
            });
        });
    });
})();
