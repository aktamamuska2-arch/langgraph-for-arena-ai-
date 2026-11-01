# Eksperimen "AI Belajar dari AI"
#
#   make dashboard   bangun ulang dashboard statis dari PLAN.md + status.json
#   make open        jalankan server lokal di http://localhost:8000
#   make verify      cek bahwa dashboard cocok dengan berkas di disk
#   make policy      lint kebijakan kegagalan (ops/policy.v1.json)
#   make calibrate   hitung ulang FWER/power/MDE aturan collapse (±80 detik, tanpa dependensi)
#   make prereg-verify  pastikan PREREGISTRATION.md tidak berubah sejak dikunci
#   make blind-status   status protokol buta P1
#   make corpus      bangun korpus guru T0 P1 (>= 2 MB) di data/p1/t0
#   make audit       audit 5 gerbang korpus T0 (keluar != 0 bila ada yang gagal)
#   make ui          antarmuka uji manusia di http://localhost:8010
#   make ui-cli      versi terminal dari antarmuka uji
#
# Semua target hanya butuh python3 (tanpa paket eksternal).
#
# BELUM ADA (baru akan dibuat pada fase P3–P6 sesuai PLAN.md):
#   make pilot    rantai mini end-to-end (200 ribu token, model 2 juta parameter)
#   make sweep    126 run eksperimen (6 arm x 3 seed x M0-M6)
#   make report   analisis + 8 grafik + laporan
# Kalau kamu menjalankannya sekarang, make akan gagal — dan itu memang benar.

PY ?= python3
PORT ?= 8000
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: dashboard open verify policy calibrate prereg prereg-verify blind-status corpus audit ui ui-cli check clean

dashboard:
	$(PY) $(HERE)dashboard/build.py

open:
	@echo "buka http://localhost:$(PORT)"
	cd $(HERE)dashboard && $(PY) -m http.server $(PORT) --bind 0.0.0.0

verify:
	@$(PY) $(HERE)dashboard/build.py >/dev/null
	@cd $(HERE) && git status --short . || true
	@echo "--- hash berkas ---"
	@cd $(HERE) && find . -type f ! -name index.html ! -path '*/__pycache__/*' -print0 \
		| sort -z | xargs -0 sha256sum

policy:
	$(PY) $(HERE)ops/policy_lint.py

calibrate:
	$(PY) $(HERE)analysis/calibration.py

corpus:
	cd $(HERE) && $(PY) -m src.p1_corpus --build

audit:
	$(PY) $(HERE)analysis/t0_audit.py

ui:
	$(PY) $(HERE)tests/ui_inspect.py --port 8010 --bind 0.0.0.0

ui-cli:
	$(PY) $(HERE)tests/ui_inspect.py --cli

prereg:
	$(PY) $(HERE)ops/prereg_lock.py --lock

prereg-verify:
	@$(PY) $(HERE)ops/prereg_lock.py --verify

blind-status:
	@$(PY) $(HERE)ops/blind.py status

check: policy prereg-verify dashboard
	@echo "OK: kebijakan lolos lint, pra-registrasi utuh, dashboard dibangun ulang"

clean:
	rm -f $(HERE)dashboard/index.html
