"""Read-only acoustic onset inventory. NO MIDI; NO segmentation decisions.
Run: python tests/diagnose_attack_candidates.py
Reads previously generated PREDESTINATI_acoustic_microstructure.csv.
"""
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
SRC = HERE / 'PREDESTINATI_acoustic_microstructure.csv'
DST = HERE / 'PREDESTINATI_attack_candidates.txt'

def main():
    with SRC.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    lines = ['OBSERVATIONAL ATTACK CANDIDATES — NOT MIDI BOUNDARIES',
             'Waveform-derived descriptors only; no threshold/classification.', '']
    for island in (2, 3, 8):
        data = [r for r in rows if int(r['island']) == island]
        if not data:
            raise ValueError(f'No acoustic microstructure data for island {island}')
        lines.append(f'ISLAND {island:02d}')
        for label, field, count in [('largest spectral changes', 'flux', 5),
                                    ('largest energy increases', 'energy_rise_db', 5),
                                    ('deepest amplitude valleys', 'rms_dbfs', 3)]:
            values = []
            for row in data:
                try:
                    v = float(row[field]); t = float(row['time_s'])
                except (ValueError, TypeError):
                    continue
                values.append((v, t, row['zone']))
            values.sort(reverse=(field != 'rms_dbfs'))
            lines.append(f'  {label}:')
            for v, t, zone in values[:count]:
                lines.append(f'    {t:.4f}s {field}={v:.3f} ({zone})')
        lines.append('  No automatic note-on/off; human/acoustic adjudication required.')
        lines.append('')
    DST.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print('Wrote', DST)

if __name__ == '__main__':
    main()
