Songwriter's Toolkit GUI - project load/save patch

Apply from /home/theea/songwriter_s_toolkit/dev/gui by copying this overlay into the repo root.

Changes:
- container inherits host timezone via /etc/localtime
- Projects root remains the explicit SONGWRITER_TOOLKIT_ROOT/Projects bind mount
- settings.json becomes authoritative GUI reload state (no legacy compatibility)
- Load button + saved-project list
- loaded projects reuse their server-side copied acapella
- Generate on an existing project asks Overwrite / Save as new project / Cancel
- Save as new copies the existing acapella and creates a new timestamped project folder
- fixes keyed-entry ID allocation after load
