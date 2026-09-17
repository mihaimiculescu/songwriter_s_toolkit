INSTALL: copy both .py files into repository tests/; no production files changed.
Run from repo root:
python tests/diagnose_53s_harmonic_reset.py --output tests/PREDESTINATI_53s_harmonic_reset_audit_FIXED.txt
python tests/diagnose_53s_selector_path.py --previous-hz 473.469
Previous-hz 473.469 is from prior synchronized frame's reported LAST F0; verify your report before passing it.
Upload both generated TXT outputs. The selector replay uses the actual local WAV and the installed candidate module.
