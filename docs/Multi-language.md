# Multi-language Support Plan

Target languages:

| Code | Language | Script | Notes |
|------|----------|--------|-------|
| en | English | Latin | Baseline |
| zh | Mandarin | CJK | Simplified Chinese |
| de | German | Latin | Strings ~30% longer than English |
| hu | Hungarian | Latin | Diacritics (ő, ű, á, é) |
| fa | Persian | Arabic | RTL — Iran's official language (Farsi) |
| ru | Russian | Cyrillic | |
| ro | Romanian | Latin | Diacritics (ă, â, î, ș, ț) |
| it | Italian | Latin | |
| fr | French | Latin | Strings moderately longer than English |
| pl | Polish | Latin | Diacritics (ą, ę, ó, ś, ź, ż) |
| cs | Czech | Latin | Diacritics (č, ď, ě, ř, š, ž) |
| es | Spanish | Latin | |
| da | Danish | Latin | Scandinavian — covers Danish |
| nb | Norwegian | Latin | Scandinavian — Bokmål; Nynorsk (nn) shares the file |
| sv | Swedish | Latin | |
| nl | Dutch | Latin | |
| pt | Portuguese | Latin | See note on pt-BR vs pt-PT below |
| ar | Arabic | Arabic | RTL — distinct from Persian, same script family |

**Two RTL languages** (Persian and Arabic) share the same directional infrastructure
but are separate translation files. Arabic has more complex ligature shaping —
Noto Sans Arabic handles both correctly.

---

## Scope

| Surface | In scope |
|---------|----------|
| Web UI — all game screens, menus, overlays, tooltips | Yes |
| Web UI — Opening Explorer, Databases, Tools, Puzzles pages | Yes |
| Install scripts (`install.sh`, `install.ps1`, `.command`, `.bat`) | Yes |
| Launcher output (`run_nmm.sh`, `run_nmm.bat`) | Yes |
| LLM commentary (Ollama prompts) | Yes — prompt language |
| AI move explanations / hints | Yes — prompt language |
| Game rules / help text | Yes |
| Docs (README, INSTALL_FROM_SOURCE, etc.) | Out of scope for v1 |

---

## Implementation Status (as of 2026-10-19)

### Completed

| Item | Status |
|------|--------|
| `web/static/i18n.js` — t(), setLanguage(), getLanguage(), saveLanguage() | ✅ Done |
| `web/static/i18n/en.json` — full baseline (~260 keys) | ✅ Done |
| `web/static/i18n/de.json` — full German translation | ✅ Done |
| `data/settings.template.json` — `"language"` field | ✅ Done |
| Language selector (`#hdr-lang`) in `index.html` header | ✅ Done |
| `index.html` — all static HTML strings wired with `data-i18n` / `data-i18n-ph` | ✅ Done |
| `game.js` — `_t()` helper; `setTurnBadge`, `updateHintButton`, slider labels | ✅ Done |
| `game.js` — all `setStatus()` call sites | ✅ Done |
| `explorer.html` — script tag + `data-i18n` on header, panels, filter chips | ✅ Done |
| `puzzles.html` — script tag + `data-i18n` on mode buttons, controls, setup panel, dialog | ✅ Done |
| `databases.html` — script tag + `data-i18n` + JS `stateLabel()` i18n + `languagechange` re-render | ✅ Done |
| `overlays.html` — new AI Overlay Guide page, all content under `data-i18n` | ✅ Done |
| `i18n.js` — added `data-i18n-title` attribute support | ✅ Done |
| ⓘ buttons in `index.html` (Overlay: label) and `explorer.html` (Nets: label) → `/overlays` | ✅ Done |

### Key i18n infrastructure patterns

**`data-i18n` attribute** (static HTML → auto-updated by `_applyDataI18n()` on every `languagechange`):
```html
<button data-i18n="nav.new_game">New Game</button>
```
For icon buttons, wrap only the text:
```html
<button>&#9658; <span data-i18n="nav.new_game">New Game</span></button>
```

**`data-i18n-ph` attribute** (for input placeholders):
```html
<input data-i18n-ph="chat.ask_ph" placeholder="Ask MillsAI…">
```

**`data-i18n-title` attribute** (for tooltip `title` attributes):
```html
<a href="/overlays" data-i18n-title="overlay.info_btn_title" title="About overlays">ⓘ</a>
```

**`_t()` in game.js / JS files** (dynamic strings set at runtime):
```js
const _t = (key, vars) => window.i18n?.t(key, vars) ?? key;
setStatus(_t("status.your_turn_place"));
setStatus(_t("status.ai_thinking", { color: colorName, elapsed: "1.2", expected: "3s" }));
```

**`window.i18n` global bridge**: i18n.js sets `window.i18n` synchronously at module load, before the async `setLanguage` auto-init. This lets non-module scripts (and scripts loaded with versioned URLs) access the same string table.

### Key namespaces in en.json / de.json

| Namespace | Coverage |
|-----------|----------|
| `nav.*` | Header nav buttons |
| `chat.*` | Chat/profile panel |
| `profile.*` | Player profile section |
| `moves.*`, `replay.*` | Moves list and replay bar |
| `opening.*` | Opening Explorer panel in game |
| `status.*` | Status bar messages (all dynamic) |
| `badge.*` | Turn/win badges |
| `btn.*` | Hint button |
| `info.*` | Game info panel |
| `newgame.*` | New Game panel |
| `setup.*` | Board setup panel |
| `paste.*` | Paste from Explorer panel |
| `tuning.*` | Personality tuning panel |
| `analysis.*` | Analysis overlay panel |
| `search.*` | Search progress panel |
| `book.*` | The Book panel |
| `bottom.*` | Bottom action bar |
| `tournament.*` | Tournament mode |
| `ava.*` | AI vs AI dialog |
| `color.*` | Color names (White / Black) |
| `weight_group.*` | Personality slider groups |
| `weight.*` | Individual personality slider labels |
| `explorer.*` | Position Explorer page |
| `puzzles.*` | NMM Puzzles page |
| `db.*` | Databases page |
| `overlay.*` | AI Overlay Guide page |

### What a new language file needs to do

1. Copy `web/static/i18n/en.json` as a starting point.
2. Translate all values (keys are never translated — they are code identifiers).
3. For `{var}` interpolation tokens (e.g. `{color}`, `{elapsed}`, `{n}`) — keep the token unchanged, translate the surrounding text.
4. Add the language code to the `supported` array in `web/static/i18n.js` (line ~92).
5. Add a `<option>` for the language to the `#hdr-lang` select in `web/templates/index.html`.
6. No backend changes required — the JSON file is served as a static asset.

### Wave 1 complete (2026-10-19)

All 9 Latin-script Wave 1 language files done — 395 keys each, JSON-valid:

| Code | Language | Status |
|------|----------|--------|
| fr | French | ✅ Done |
| es | Spanish (neutral) | ✅ Done |
| it | Italian | ✅ Done |
| nl | Dutch | ✅ Done |
| pt | Portuguese (PT-PT) | ✅ Done |
| sv | Swedish | ✅ Done |
| da | Danish | ✅ Done |
| nb | Norwegian Bokmål | ✅ Done |
| ro | Romanian (correct ș/ț diacritics) | ✅ Done |

### Wave 2 complete (2026-10-19)

All 7 Wave 2 language files done — 395 keys each, JSON-valid:

| Code | Language | Notes |
|------|----------|-------|
| zh | Simplified Chinese | Lazy Noto Sans SC font injection |
| hu | Hungarian | malom (game term) / Malom (database) disambiguated |
| fa | Persian/Farsi | RTL; correct ک/ی (not Arabic lookalikes) |
| ru | Russian | Cyrillic; Noto Sans lazy injection |
| pl | Polish | Full diacritics ą/ć/ę/ł/ń/ó/ś/ź/ż |
| cs | Czech | Full diacritics á/č/ď/é/ě/í/ň/ř/š/ť/ú/ů/ý/ž |
| ar | Arabic | RTL; MSA; Eastern Arabic numerals in prose |

RTL infrastructure also added in this wave:
- `web/static/rtl.css` — flex-row reversals, text-align, input direction, font rules
- `web/static/i18n.js` — lazy Google Font injection for zh/ar/fa/ru; `dir=rtl` + `.rtl` class already wired
- `rtl.css` linked in all six templates

### Still to do

- [ ] `explorer.js` — dynamic strings generated in JS (move list labels, overlay text)
- [ ] Phase 3 — install script localisation
- [ ] Phase 4 — LLM commentary language instruction

---

## Phase 1 — Infrastructure

### 1.1 Translation file format

One JSON file per language under `web/static/i18n/`:

```
web/static/i18n/
    en.json      ← baseline (source of truth)
    zh.json  de.json  hu.json  fa.json  ru.json
    ro.json  it.json  fr.json  pl.json  cs.json
    es.json  da.json  nb.json  sv.json  nl.json
    pt.json  ar.json
```

Each file is a flat key → string map. Keys are dot-namespaced by screen:

```json
{
  "nav.play":           "Play",
  "nav.databases":      "Databases",
  "nav.openings":       "Openings",
  "game.your_turn":     "Your turn",
  "game.ai_thinking":   "AI is thinking…",
  "settings.language":  "Language",
  "db.download":        "Download",
  ...
}
```

Plurals and interpolation use a minimal `{count}` / `{name}` convention —
no full ICU message format needed at this scale.

### 1.2 JS i18n module

New file `web/static/i18n.js`:

- Exports `t(key, vars)` — looks up key in loaded strings, interpolates `{var}` tokens, falls back to English if key missing.
- Exports `setLanguage(code)` — fetches `/static/i18n/{code}.json`, caches it, fires a `languagechange` CustomEvent on `document`.
- Exports `getLanguage()` — returns active code.
- On load, resolves language from (in order): `localStorage.nmm_lang` → `settings.language` → `navigator.language` prefix → `"en"`.

All existing templates import `i18n.js` and replace hardcoded strings with `t("key")` calls. The `languagechange` event triggers a lightweight re-render of text nodes without a full page reload.

### 1.3 RTL support (Persian and Arabic)

- `<html>` gets `dir="rtl"` and `class="rtl"` when `lang === "fa"` or `lang === "ar"`.
- A `web/static/rtl.css` stylesheet handles directional overrides:
  - flex-direction mirrors, float swaps, margin/padding L↔R swaps for layout elements.
  - Board overlays and SVG arrows are excluded — game geometry is fixed.
  - The sidebar panel slides from the right instead of left.
- Language switcher sets `dir` on `<html>` immediately, before the string reload completes, so layout doesn't flash.
- Arabic requires more complex glyph shaping than Persian (contextual letter forms, ligatures) — Noto Sans Arabic handles both scripts; no separate font needed.

### 1.4 Font loading

Add `<link>` preloads in `base.html` (or equivalent) for:

| Language | Font | Source |
|----------|------|--------|
| Mandarin | Noto Sans SC | Google Fonts (self-hosted copy in `web/static/fonts/`) |
| Persian, Arabic | Noto Sans Arabic | same — one font covers both scripts |
| Russian | Noto Sans (Cyrillic subset) | same |
| All Latin languages | System fonts sufficient; Noto Sans Latin as fallback | — |

Latin languages (German, French, Spanish, Italian, Portuguese, Dutch, Romanian,
Hungarian, Polish, Czech, Swedish, Danish, Norwegian) all use standard Latin +
extended Latin Unicode — no special font loading required beyond the system stack.

Self-host fonts to avoid external network dependency at game runtime. Add to
`web/static/fonts/` and reference via `@font-face` in `web/static/style.css`.

### 1.5 Language setting persistence

Add `"language": "en"` to `data/settings.template.json`.
The `/api/settings` endpoint already reads/writes settings — no backend change needed.
The JS i18n module reads `settings.language` on page load and writes back on change.

### 1.6 Language switcher UI

Add a small flag/code dropdown to the page header (right side, next to the existing
icon buttons). Selecting a language:
1. Writes to `localStorage.nmm_lang` immediately (fast).
2. POSTs `{"language": code}` to `/api/settings` (persisted across sessions).
3. Fires `setLanguage(code)`.

---

## Phase 2 — Web UI translation

### 2.1 String extraction

Run a one-time extraction pass over all templates and JS files to identify every
hardcoded user-visible string. Produce `en.json` as the baseline. Tool:
`tools/extract_i18n_strings.py` — walks `web/templates/` and `web/static/*.js`,
finds string literals matching known UI patterns, outputs a sorted `en.json`.

The extractor is a guide, not exhaustive — manual review pass required after.

### 2.2 Translation priority order

| Priority | Screens |
|----------|---------|
| 1 | Game board UI, turn indicators, win/lose/draw messages |
| 2 | Settings panel, difficulty labels, AI personality names |
| 3 | Opening Explorer labels and move notations |
| 4 | Databases page (bundle names, download status) |
| 5 | Puzzles page |
| 6 | Tools page |

### 2.3 Move notation

Standard Nine Men's Morris algebraic notation (`a1`, `d4-e4`, `b2xd2`) is language-
neutral — no translation required. Phase/action labels ("Place", "Move", "Fly",
"Mill formed") are translated.

### 2.4 Numbers and formatting

- Elo ratings, game counters: use `Intl.NumberFormat(langCode)` for locale-aware
  separators.
- Dates in game history: use `Intl.DateTimeFormat(langCode)`.
- No currency involved.

---

## Phase 3 — Install script localisation

### 3.1 Language detection

**Linux / macOS (`install.sh`, `run_nmm.sh`):**

```bash
_lang="${LANG:-en}"
case "$_lang" in
    zh*)       _L="zh" ;;
    de*)       _L="de" ;;
    hu*)       _L="hu" ;;
    fa*)       _L="fa" ;;
    ar*)       _L="ar" ;;
    ru*)       _L="ru" ;;
    ro*)       _L="ro" ;;
    it*)       _L="it" ;;
    fr*)       _L="fr" ;;
    pl*)       _L="pl" ;;
    cs*)       _L="cs" ;;
    es*)       _L="es" ;;
    da*)       _L="da" ;;
    nb*|no*)   _L="nb" ;;
    sv*)       _L="sv" ;;
    nl*)       _L="nl" ;;
    pt*)       _L="pt" ;;
    *)         _L="en" ;;
esac
source "$NMM_DIR/install_i18n/${_L}.sh"
```

**Windows (`install.ps1`, `run_nmm.bat`):**

```powershell
$ui = [System.Globalization.CultureInfo]::CurrentUICulture.TwoLetterISOLanguageName
$supported = @("zh","de","hu","fa","ar","ru","ro","it","fr","pl","cs","es","da","nb","sv","nl","pt")
$lang = if ($supported -contains $ui) { $ui } else { "en" }
. "$NMM_DIR\install_i18n\$lang.ps1"
```

### 3.2 Message files

One message file per language per platform:

```
install_i18n/
    en.sh   de.sh   hu.sh   fa.sh   ar.sh   ru.sh   zh.sh
    ro.sh   it.sh   fr.sh   pl.sh   cs.sh   es.sh
    da.sh   nb.sh   sv.sh   nl.sh   pt.sh
    en.ps1  de.ps1  hu.ps1  fa.ps1  ar.ps1  ru.ps1  zh.ps1
    ro.ps1  it.ps1  fr.ps1  pl.ps1  cs.ps1  es.ps1
    da.ps1  nb.ps1  sv.ps1  nl.ps1  pt.ps1
```

Each file defines shell functions / variables used by the installer:
`msg_welcome`, `msg_python_ok`, `msg_venv_create`, `msg_done`, etc.
English `.sh` / `.ps1` files are the source; other languages are translations of those.

### 3.3 Terminal encoding

- Linux/macOS: UTF-8 is standard; Mandarin/Persian/Russian characters display
  correctly in most modern terminals.
- Windows: PowerShell 5.1 defaults to the system code page. Add
  `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` at the top of each
  `.ps1` message file and `chcp 65001` at the top of `.bat` files.
- Persian and Arabic in a Windows terminal are functional but RTL rendering in
  `cmd.exe` / PowerShell is poor — acceptably readable, not visually correct.
  Note this in the Persian and Arabic message files. All other languages display
  correctly on Windows with `chcp 65001`.

---

## Phase 4 — LLM commentary localisation

The Ollama commentary system constructs prompts in `ai/ingame_commentator.py`
(and related). When a language other than English is active:

1. Append a language instruction to the system prompt:
   `"Respond entirely in {language_name}. Do not use English."`
2. The language is read from `settings.language` at commentary request time.
3. No translation of the prompt templates themselves is needed — LLMs handle
   language instruction well.

For Persian and Arabic, note in the settings UI that RTL commentary text in
the chat panel will display correctly (the browser handles Unicode BiDi automatically).

---

## Phase 5 — Testing checklist

Per language, before marking as complete:

- [ ] All Phase 2 screens render without layout overflow (risk languages: Mandarin — CJK wider; German / French — strings ~30% longer; Arabic — RTL + wide glyphs)
- [ ] RTL layout correct in Persian and Arabic — no element collisions, scrollbars on correct side
- [ ] Install script: message file sourced correctly on locale detection for each platform
- [ ] Windows UTF-8 (`chcp 65001`): Cyrillic, CJK, Arabic, and Latin diacritics all visible in terminal
- [ ] LLM commentary responds in target language
- [ ] Language preference persists across server restarts
- [ ] Fallback to English works when a key is missing in a translation file
- [ ] Norwegian (nb) file also serves Nynorsk (nn) locale gracefully
- [ ] Portuguese file confirmed appropriate for both pt-PT and pt-BR speakers, or split into pt-PT / pt-BR if divergence is significant

---

## File structure summary

```
web/static/
    i18n.js                    ← JS i18n module (new)
    i18n/
        en.json  zh.json  de.json  hu.json  fa.json  ar.json  ru.json
        ro.json  it.json  fr.json  pl.json  cs.json  es.json
        da.json  nb.json  sv.json  nl.json  pt.json
    rtl.css                    ← RTL layout overrides for fa + ar (new)
    fonts/
        NotoSansSC-Regular.woff2        ← Mandarin
        NotoSansArabic-Regular.woff2    ← Persian + Arabic
        NotoSans-Regular.woff2          ← Cyrillic fallback (Russian)

install_i18n/
    en.sh   de.sh   hu.sh   fa.sh   ar.sh   ru.sh   zh.sh
    ro.sh   it.sh   fr.sh   pl.sh   cs.sh   es.sh
    da.sh   nb.sh   sv.sh   nl.sh   pt.sh
    en.ps1  de.ps1  hu.ps1  fa.ps1  ar.ps1  ru.ps1  zh.ps1
    ro.ps1  it.ps1  fr.ps1  pl.ps1  cs.ps1  es.ps1
    da.ps1  nb.ps1  sv.ps1  nl.ps1  pt.ps1

tools/
    extract_i18n_strings.py    ← one-time string extraction helper (new)

data/
    settings.template.json     ← add "language": "en"
```

---

## Implementation order

1. Phase 1 infrastructure (i18n.js, RTL CSS, font loading, settings field, language switcher)
2. `en.json` string extraction and baseline
3. **Wave 1 — Latin, low risk:** German, French, Spanish, Italian, Portuguese, Dutch, Romanian, Swedish, Danish, Norwegian (10 languages — share font stack, no layout surprises beyond string length)
4. **Wave 2 — Latin + diacritics:** Polish, Czech, Hungarian (3 languages — verify diacritic rendering)
5. **Wave 3 — Non-Latin scripts:** Russian / Cyrillic (tests Noto Sans Cyrillic font fallback)
6. **Wave 4 — CJK:** Mandarin (tests Noto Sans SC loading, wider character layout)
7. **Wave 5 — RTL:** Persian, then Arabic (full RTL pipeline; Arabic adds ligature complexity on top of Persian)
8. Install script message files (all languages in one pass, using existing translations)
9. LLM commentary language instruction (one-line change, applies to all languages)
10. Phase 5 testing pass per language, RTL languages last

---

## Open questions

- **Translation source**: machine-translated (DeepL/GPT) then human-reviewed, or
  native-speaker contributed? Affects quality and timeline significantly. GPT-4 is
  adequate for Wave 1 Latin languages; native review is more important for Arabic,
  Persian, and Mandarin.
- **Portuguese split**: pt-BR (Brazilian, ~215M speakers) and pt-PT (European, ~10M)
  differ enough in vocabulary that a single `pt.json` may feel off to one group.
  Consider `pt-BR.json` + `pt-PT.json`, or ship one and note the limitation.
- **Scandinavian shared file**: Danish and Norwegian Bokmål are closely related —
  a single `nb.json` with a note in the UI ("Norwegian / Dansk") is pragmatic for v1.
  Swedish and Danish are distinct enough to warrant separate files (`sv.json`, `da.json`).
- **AI personality names**: translate descriptively or keep English as proper nouns?
  Recommend keeping English names (e.g. "Aggressive", "Sentinel") as they function
  as labels, but translate the description text.
- **Opening names** (Mill Rush, Central Anchor, etc.): recommend keeping English
  as canonical labels — they appear in the opening book data and changing them
  would require a data migration. Translate surrounding UI text only.
- **Arabic dialect**: Modern Standard Arabic (MSA / فصحى) is understood across all
  Arabic-speaking regions and is the right choice for a game UI. Regional dialects
  would exclude other Arabic speakers.
