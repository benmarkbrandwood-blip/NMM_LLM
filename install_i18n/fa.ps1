# install_i18n/fa.ps1 — Persian (Farsi) messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- نصب‌کننده     |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "پوشه نصب: {0}"
$P_PATH_ERROR       = "مسیر نصب شامل '[' یا ']' است که PowerShell آن را بد مدیریت می‌کند. پوشه را به مسیر ساده‌تری (مثلاً C:\NMM) منتقل کرده و مجدداً اجرا کنید."

$P_PYTHON_SEARCH    = 'جستجوی Python نسخه ۳.۱۰ یا جدیدتر...'
$P_PYTHON_FOUND     = "استفاده از Python {0}.{1} از طریق '{2}'."
$P_PYTHON_OLD       = "'{0}' نسخه Python {1}.{2} است -- نیاز به ۳.۱۰+؛ تلاش برای بعدی."
$P_PYTHON_MISSING   = 'Python نسخه ۳.۱۰ یا جدیدتر یافت نشد.'
$P_PYTHON_URL       = 'آن را از https://www.python.org/downloads/ نصب کنید'
$P_PYTHON_PATH      = "مهم: هنگام نصب گزینه 'Add python.exe to PATH' را علامت بزنید."
$P_PYTHON_FAIL      = 'Python نسخه ۳.۱۰ یا بالاتر الزامی است.'

$P_VENV_EXISTS      = 'محیط مجازی .venv موجود یافت شد -- استفاده مجدد.'
$P_VENV_CREATE      = 'در حال ایجاد محیط مجازی در .venv...'
$P_VENV_FAIL        = "ایجاد محیط مجازی ممکن نشد. مطمئن شوید ماژول Python 'venv' در دسترس است."

$P_PIP_UPGRADE      = 'به‌روزرسانی pip / setuptools / wheel...'
$P_PIP_WARN         = 'به‌روزرسانی pip کد خروج غیرصفر برگرداند؛ ادامه می‌دهیم.'

$P_REQS_INSTALL     = 'نصب پیش‌نیازهای Python (ممکن است چند دقیقه طول بکشد)...'
$P_REQS_CPP         = "در ویندوز، chromadb گاهی به 'Microsoft C++ Build Tools' نیاز دارد."
$P_REQS_CPP_URL     = 'از https://visualstudio.microsoft.com/visual-cpp-build-tools/ نصب کنید'
$P_REQS_CPP_RERUN   = 'سپس install.bat را مجدداً اجرا کنید.'
$P_REQS_FAIL        = 'نصب وابستگی‌های Python ناموفق بود.'
$P_REQS_DONE        = 'بسته‌های Python نصب شدند.'

$P_OLLAMA_SKIP      = 'Ollama رد شد (پرچم -NoOllama تنظیم شده). ویژگی‌های LLM غیرفعال می‌شوند.'
$P_OLLAMA_AUTO      = 'نصب خودکار Ollama (پرچم -Yes تنظیم شده).'
$P_OLLAMA_PROMPT1   = 'Ollama تفسیر هوش مصنوعی و تحلیل حرکت با LLM را فعال می‌کند.'
$P_OLLAMA_PROMPT2   = "مدل پیش‌فرض ({0}) حدود ۵ گیگابایت است و نیاز به دانلود دارد."
$P_OLLAMA_YN        = '[NMM] آیا Ollama را برای ویژگی‌های LLM نصب کنید؟ [Y/n]'
$P_OLLAMA_SKIP2     = 'Ollama رد شد. دوباره install.ps1 را اجرا کنید (بدون -NoOllama) تا بعداً اضافه شود.'
$P_OLLAMA_EXISTS    = "Ollama از قبل نصب شده ({0})."
$P_OLLAMA_DOWNLOAD  = 'دانلود نصب‌کننده Ollama...'
$P_OLLAMA_DL_FAIL   = "دانلود خودکار نصب‌کننده Ollama ممکن نشد: {0}"
$P_OLLAMA_DL_MANUAL = 'آن را به‌صورت دستی از https://ollama.com نصب کرده، سپس install.bat را مجدداً اجرا کنید.'
$P_OLLAMA_RUN       = 'اجرای نصب‌کننده Ollama (دستورالعمل‌ها را دنبال کنید)...'
$P_OLLAMA_RUN_FAIL  = "نصب‌کننده Ollama تکمیل نشد: {0}"
$P_OLLAMA_PATH      = 'پس از نصب، Ollama در PATH یافت نشد.'
$P_OLLAMA_PATH_HINT = "ترمینال را مجدداً راه‌اندازی کنید، سپس اجرا کنید:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama نصب شد.'
$P_OLLAMA_START     = 'راه‌اندازی سرویس Ollama...'
$P_OLLAMA_RUNNING   = 'سرویس Ollama در حال اجرا است.'
$P_OLLAMA_TIMEOUT   = 'سرویس Ollama به موقع پاسخ نداد. آن را به‌صورت دستی اجرا کنید: ollama serve'

$P_MODEL_CHECK      = "بررسی مدل '{0}'..."
$P_MODEL_EXISTS     = "مدل '{0}' از قبل موجود است."
$P_MODEL_PULL       = "دانلود '{0}' -- ممکن است چند دقیقه طول بکشد..."
$P_MODEL_PULL_FAIL  = "دانلود مدل ناموفق بود. بعداً به‌صورت دستی اجرا کنید: ollama pull {0}"
$P_MODEL_READY      = "مدل '{0}' آماده است."

$P_DIRS_READY       = 'پوشه‌های داده آماده‌اند.'
$P_SETTINGS_CREATED = 'تنظیمات پیش‌فرض در data\settings.json ایجاد شد.'
$P_DB_ERROR         = 'راه‌اندازی پایگاه داده با خطا مواجه شد -- در بازی از /databases ادامه دهید.'
$P_DB_MISSING       = 'tools\install_dbs.py یافت نشد -- دانلود پایگاه داده رد شد.'

$P_DONE             = 'نصب کامل شد!'
$P_START_HINT       = 'برای شروع بازی:'
$P_START_CMD        = '    .\run_nmm.bat'
