// i18n module — t(), setLanguage(), getLanguage()
// Loads /static/i18n/{code}.json, fires 'languagechange' on document.

let _strings = {};
let _lang = 'en';

export function t(key, vars) {
  let s = _strings[key] ?? key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      s = s.replaceAll(`{${k}}`, v);
    }
  }
  return s;
}

export function getLanguage() {
  return _lang;
}

// Lazy-load Google Fonts for languages that need non-system scripts.
const _FONT_URLS = {
  zh: 'https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;600&display=swap',
  ar: 'https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;600&display=swap',
  fa: 'https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;600&display=swap',
  ru: 'https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600&display=swap',
};
const _loadedFonts = new Set();
function _injectFont(code) {
  const url = _FONT_URLS[code];
  if (!url || _loadedFonts.has(code) || typeof document === 'undefined') return;
  _loadedFonts.add(code);
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = url;
  document.head.appendChild(link);
}

export async function setLanguage(code) {
  if (code !== 'en') {
    try {
      const r = await fetch(`/static/i18n/${code}.json`);
      if (!r.ok) throw new Error(r.status);
      _strings = await r.json();
    } catch {
      _strings = {};
    }
  } else {
    _strings = {};
  }

  // English baseline: load en.json so fallback keys are populated
  if (code !== 'en') {
    try {
      const r = await fetch('/static/i18n/en.json');
      if (r.ok) {
        const base = await r.json();
        _strings = Object.assign({}, base, _strings);
      }
    } catch { /* ignore */ }
  } else {
    try {
      const r = await fetch('/static/i18n/en.json');
      if (r.ok) _strings = await r.json();
    } catch { _strings = {}; }
  }

  _lang = code;

  const isRTL = code === 'fa' || code === 'ar';
  document.documentElement.lang = code;
  document.documentElement.dir = isRTL ? 'rtl' : 'ltr';
  document.documentElement.classList.toggle('rtl', isRTL);
  _injectFont(code);

  _applyDataI18n();

  document.dispatchEvent(new CustomEvent('languagechange', { detail: { lang: code } }));
}

function _applyDataI18n() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (key) el.textContent = t(key);
  });
  document.querySelectorAll('[data-i18n-ph]').forEach(el => {
    const key = el.dataset.i18nPh;
    if (key) el.placeholder = t(key);
  });
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    const key = el.dataset.i18nTitle;
    if (key) el.title = t(key);
  });
}

// Persist language
export function saveLanguage(code) {
  localStorage.setItem('nmm_lang', code);
  fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ language: code })
  }).catch(() => {});
}

// Expose synchronously so game.js can call _t() even before auto-init resolves
window.i18n = { t, setLanguage, getLanguage, saveLanguage };

// Auto-init on module load
(async () => {
  // Priority: localStorage → navigator.language prefix → 'en'
  const stored = localStorage.getItem('nmm_lang');
  let code = stored || navigator.language?.split('-')[0] || 'en';

  const supported = [
    'en','zh','de','hu','fa','ru','ro','it','fr',
    'pl','cs','es','da','nb','sv','nl','pt','ar'
  ];
  if (!supported.includes(code)) code = 'en';

  await setLanguage(code);
})();
