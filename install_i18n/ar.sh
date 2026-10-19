#!/usr/bin/env bash
# install_i18n/ar.sh — Arabic messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="جارٍ التحقق من Python..."
I_PYTHON_OK="Python %s — تم"
I_PYTHON_MISSING="لم يُعثر على Python 3. قم بتثبيت python3 ثم أعد المحاولة."
I_PYTHON_OLD="يلزم Python 3.10 أو أحدث (تم العثور على: %s)."

I_VENV_EXISTS="تم العثور على بيئة venv موجودة في .venv — يتم تخطي الإنشاء."
I_VENV_CREATE="جارٍ إنشاء بيئة افتراضية في .venv ..."
I_PIP_UPGRADE="جارٍ ترقية pip..."
I_REQS_INSTALL="جارٍ تثبيت متطلبات Python..."
I_REQS_DONE="تم تثبيت حزم Python."

I_OLLAMA_EXISTS="Ollama مثبَّت بالفعل (%s)."
I_OLLAMA_INSTALL="جارٍ تثبيت Ollama..."
I_BREW_MISSING="لم يُعثر على Homebrew. قم بتثبيت Ollama يدويًا من https://ollama.com/download/mac"
I_BREW_RERUN="ثم أعد تشغيل هذا السكريبت."
I_OLLAMA_DONE="تم تثبيت Ollama."
I_OLLAMA_START="جارٍ تشغيل خدمة Ollama..."
I_OLLAMA_STARTED="تم تشغيل خدمة Ollama (pid %s)."
I_OLLAMA_RUNNING="خدمة Ollama تعمل بالفعل."
I_OLLAMA_TIMEOUT="لم تبدأ خدمة Ollama في الوقت المحدد."

I_MODEL_CHECK="جارٍ التحقق من النموذج '%s'..."
I_MODEL_EXISTS="النموذج '%s' موجود بالفعل."
I_MODEL_PULL="جارٍ تنزيل '%s' — قد يستغرق ذلك عدة دقائق..."
I_MODEL_READY="النموذج '%s' جاهز."

I_DIRS_READY="مجلدات البيانات جاهزة."
I_SETTINGS_CREATED="تم إنشاء الإعدادات الافتراضية في data/settings.json."
I_DB_ERROR="واجه إعداد قاعدة البيانات خطأً — تابع من /databases في اللعبة."
I_DB_MISSING="لم يُعثر على tools/install_dbs.py — يتم تخطي تنزيل قاعدة البيانات."

I_DONE="اكتمل التثبيت!"
I_START_HINT="لبدء اللعبة:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="لم يُعثر على البيئة الافتراضية. شغّل ./install.sh أولًا."
R_NO_UVICORN="لم يُعثر على uvicorn في venv. شغّل ./install.sh أولًا."
R_OLLAMA_START="Ollama لا يعمل — جارٍ التشغيل في الخلفية..."
R_OLLAMA_TIMEOUT="لم يستجب Ollama — سيتم تعطيل ميزات LLM."
R_PORT_BUSY="المنفذ %s قيد الاستخدام بالفعل. جارٍ تجربة المنفذ 8080..."
R_SHUTDOWN="جارٍ الإيقاف..."
R_NO_CKPT="لم يُعثر على نقاط تفتيش نموذج الذكاء الاصطناعي — ستكون ميزات الذكاء الاصطناعي محدودة."
R_DL_CKPT="قم بتنزيلها عبر ./install.sh أو افتح /databases في اللعبة."
R_STARTING="جارٍ تشغيل Nine Men's Morris على %s ..."
R_WAITING="جارٍ الانتظار حتى يكون الخادم جاهزًا..."
R_SERVER_SLOW="استغرق الخادم وقتًا طويلًا للاستجابة — سيتم فتح المتصفح على أي حال."
R_OPEN_BROWSER="جارٍ فتح المتصفح على %s"
R_NO_BROWSER="تعذّر الكشف عن متصفح. افتح %s يدويًا."
R_RUNNING="Nine Men's Morris يعمل على %s"
R_STOP_HINT="اضغط Ctrl-C للإيقاف."
