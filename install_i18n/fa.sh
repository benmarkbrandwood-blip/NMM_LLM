#!/usr/bin/env bash
# install_i18n/fa.sh — Persian messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="در حال بررسی Python..."
I_PYTHON_OK="Python %s — موفق"
I_PYTHON_MISSING="Python 3 یافت نشد. python3 را نصب کنید و دوباره امتحان کنید."
I_PYTHON_OLD="Python 3.10+ مورد نیاز است (یافت شد: %s)."

I_VENV_EXISTS="محیط مجازی موجود در .venv یافت شد — از ایجاد مجدد صرف‌نظر می‌شود."
I_VENV_CREATE="در حال ایجاد محیط مجازی در .venv ..."
I_PIP_UPGRADE="در حال ارتقاء pip..."
I_REQS_INSTALL="در حال نصب وابستگی‌های Python..."
I_REQS_DONE="بسته‌های Python نصب شدند."

I_OLLAMA_EXISTS="Ollama از قبل نصب شده است (%s)."
I_OLLAMA_INSTALL="در حال نصب Ollama..."
I_BREW_MISSING="Homebrew یافت نشد. Ollama را به صورت دستی از https://ollama.com/download/mac نصب کنید"
I_BREW_RERUN="سپس این اسکریپت را دوباره اجرا کنید."
I_OLLAMA_DONE="Ollama نصب شد."
I_OLLAMA_START="در حال راه‌اندازی سرویس Ollama..."
I_OLLAMA_STARTED="سرویس Ollama راه‌اندازی شد (pid %s)."
I_OLLAMA_RUNNING="سرویس Ollama در حال اجرا است."
I_OLLAMA_TIMEOUT="سرویس Ollama به موقع راه‌اندازی نشد."

I_MODEL_CHECK="در حال بررسی مدل '%s'..."
I_MODEL_EXISTS="مدل '%s' از قبل موجود است."
I_MODEL_PULL="در حال دانلود '%s' — این ممکن است چند دقیقه طول بکشد..."
I_MODEL_READY="مدل '%s' آماده است."

I_DIRS_READY="پوشه‌های داده آماده هستند."
I_SETTINGS_CREATED="تنظیمات پیش‌فرض در data/settings.json ایجاد شد."
I_DB_ERROR="راه‌اندازی پایگاه داده با خطا مواجه شد — از /databases در بازی ادامه دهید."
I_DB_MISSING="tools/install_dbs.py یافت نشد — از دانلود پایگاه داده صرف‌نظر می‌شود."

I_DONE="نصب کامل شد!"
I_START_HINT="برای شروع بازی:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="محیط مجازی یافت نشد. ابتدا ./install.sh را اجرا کنید."
R_NO_UVICORN="uvicorn در venv یافت نشد. ابتدا ./install.sh را اجرا کنید."
R_OLLAMA_START="Ollama در حال اجرا نیست — در حال راه‌اندازی در پس‌زمینه..."
R_OLLAMA_TIMEOUT="Ollama پاسخ نداد — قابلیت‌های LLM غیرفعال خواهند شد."
R_PORT_BUSY="پورت %s در حال استفاده است. در حال تلاش با پورت 8080..."
R_SHUTDOWN="در حال خاموش شدن..."
R_NO_CKPT="نقاط بازیابی مدل هوش مصنوعی یافت نشدند — قابلیت‌های هوش مصنوعی محدود خواهند بود."
R_DL_CKPT="آن‌ها را از طریق ./install.sh دانلود کنید یا /databases را در بازی باز کنید."
R_STARTING="در حال راه‌اندازی Nine Men's Morris در %s ..."
R_WAITING="در انتظار آماده شدن سرور..."
R_SERVER_SLOW="سرور دیر پاسخ داد — مرورگر به هر حال باز می‌شود."
R_OPEN_BROWSER="در حال باز کردن مرورگر در %s"
R_NO_BROWSER="نتوانستم مرورگری شناسایی کنم. %s را به صورت دستی باز کنید."
R_RUNNING="Nine Men's Morris در %s در حال اجرا است"
R_STOP_HINT="برای توقف Ctrl-C را فشار دهید."
