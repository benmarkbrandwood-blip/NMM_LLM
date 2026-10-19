# install_i18n/ar.ps1 — Arabic messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- مثبّت         |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "مجلد التثبيت: {0}"
$P_PATH_ERROR       = "مسار التثبيت يحتوي على '[' أو ']' مما يسبب مشكلات في PowerShell. انقل المجلد إلى مسار أبسط (مثل C:\NMM) وأعد التشغيل."

$P_PYTHON_SEARCH    = 'البحث عن Python 3.10 أو أحدث...'
$P_PYTHON_FOUND     = "استخدام Python {0}.{1} عبر '{2}'."
$P_PYTHON_OLD       = "'{0}' هو Python {1}.{2} -- مطلوب 3.10+؛ جارٍ تجربة التالي."
$P_PYTHON_MISSING   = 'لم يُعثر على Python 3.10 أو أحدث.'
$P_PYTHON_URL       = 'قم بتثبيته من https://www.python.org/downloads/'
$P_PYTHON_PATH      = "مهم: ضع علامة على 'Add python.exe to PATH' أثناء التثبيت."
$P_PYTHON_FAIL      = 'Python 3.10 أو أحدث مطلوب.'

$P_VENV_EXISTS      = 'عُثر على .venv موجود -- إعادة استخدامه.'
$P_VENV_CREATE      = 'إنشاء بيئة افتراضية في .venv...'
$P_VENV_FAIL        = "تعذّر إنشاء البيئة الافتراضية. تأكد من توفر وحدة Python 'venv'."

$P_PIP_UPGRADE      = 'تحديث pip / setuptools / wheel...'
$P_PIP_WARN         = 'أعاد تحديث pip رمز خروج غير صفري؛ المتابعة على أي حال.'

$P_REQS_INSTALL     = 'تثبيت متطلبات Python (قد يستغرق بضع دقائق)...'
$P_REQS_CPP         = "في Windows، يحتاج chromadb أحياناً إلى 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'قم بالتثبيت من https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'ثم أعد تشغيل install.bat.'
$P_REQS_FAIL        = 'فشل تثبيت تبعيات Python.'
$P_REQS_DONE        = 'تم تثبيت حزم Python.'

$P_OLLAMA_SKIP      = 'تخطي Ollama (تم تعيين علامة -NoOllama). ستُعطَّل ميزات LLM.'
$P_OLLAMA_AUTO      = 'تثبيت Ollama تلقائياً (تم تعيين علامة -Yes).'
$P_OLLAMA_PROMPT1   = 'يتيح Ollama التعليق بالذكاء الاصطناعي وتحليل الحركات بـ LLM.'
$P_OLLAMA_PROMPT2   = "النموذج الافتراضي ({0}) حجمه ~5 جيجابايت ويتطلب التنزيل."
$P_OLLAMA_YN        = '[NMM] هل تريد تثبيت Ollama لميزات LLM؟ [Y/n]'
$P_OLLAMA_SKIP2     = 'تم تخطي Ollama. أعد تشغيل install.ps1 (بدون -NoOllama) لإضافته لاحقاً.'
$P_OLLAMA_EXISTS    = "Ollama مثبّت بالفعل ({0})."
$P_OLLAMA_DOWNLOAD  = 'تنزيل مثبّت Ollama...'
$P_OLLAMA_DL_FAIL   = "تعذّر تنزيل مثبّت Ollama تلقائياً: {0}"
$P_OLLAMA_DL_MANUAL = 'قم بالتثبيت يدوياً من https://ollama.com ثم أعد تشغيل install.bat.'
$P_OLLAMA_RUN       = 'تشغيل مثبّت Ollama (اتبع التعليمات)...'
$P_OLLAMA_RUN_FAIL  = "لم يكتمل مثبّت Ollama: {0}"
$P_OLLAMA_PATH      = 'لم يُعثر على Ollama في PATH بعد التثبيت.'
$P_OLLAMA_PATH_HINT = "أعد تشغيل الطرفية ثم نفّذ:  ollama pull {0}"
$P_OLLAMA_DONE      = 'تم تثبيت Ollama.'
$P_OLLAMA_START     = 'بدء تشغيل خدمة Ollama...'
$P_OLLAMA_RUNNING   = 'خدمة Ollama تعمل بالفعل.'
$P_OLLAMA_TIMEOUT   = 'لم تستجب خدمة Ollama في الوقت المحدد. شغّلها يدوياً: ollama serve'

$P_MODEL_CHECK      = "التحقق من النموذج '{0}'..."
$P_MODEL_EXISTS     = "النموذج '{0}' موجود بالفعل."
$P_MODEL_PULL       = "تنزيل '{0}' -- قد يستغرق عدة دقائق..."
$P_MODEL_PULL_FAIL  = "فشل تنزيل النموذج. نفّذه يدوياً لاحقاً: ollama pull {0}"
$P_MODEL_READY      = "النموذج '{0}' جاهز."

$P_DIRS_READY       = 'مجلدات البيانات جاهزة.'
$P_SETTINGS_CREATED = 'تم إنشاء الإعدادات الافتراضية في data\settings.json.'
$P_DB_ERROR         = 'واجه إعداد قاعدة البيانات خطأ -- تابع عبر /databases في اللعبة.'
$P_DB_MISSING       = 'tools\install_dbs.py غير موجود -- تم تخطي تنزيل قاعدة البيانات.'

$P_DONE             = 'اكتمل التثبيت!'
$P_START_HINT       = 'لتشغيل اللعبة:'
$P_START_CMD        = '    .\run_nmm.bat'
