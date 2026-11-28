"""Check registered PDF numeric claims against small, committed CSV artifacts.

No model imports, fitting, flight records or holdout inputs. Protocol constants and
figure pixels are outside the numeric claim manifest's explicitly documented scope.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'Pilot', 'Full-development validation', 'Held-out test'}


def artifact_path(root, relative):
    path = (root / relative).resolve()
    allowed = [root / 'reports/tables', root / 'reports/p5_phase_b_pilot']
    if path.suffix != '.csv' or not any(path.is_relative_to(p.resolve()) for p in allowed):
        raise ValueError(f'Not an approved report CSV: {relative}')
    return path


def expected_value(spec, root):
    path = artifact_path(root, spec['csv'])
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows if all(r.get(k) == str(v) for k, v in spec.get('where', {}).items())]
    if not rows:
        raise ValueError(f'Empty selector: {spec}')
    op = spec.get('op', 'single')
    if op == 'count':
        value = len(rows)
    else:
        values = [float(r[spec['column']]) for r in rows]
        if op == 'single':
            if len(values) != 1:
                raise ValueError(f'Nonunique selector: {spec}')
            value = values[0]
        elif op == 'mean':
            value = fmean(values)
        elif op == 'sum':
            value = sum(values)
        elif op == 'min':
            value = min(values)
        elif op == 'max':
            value = max(values)
        elif op == 'ratio':
            value = sum(values) / sum(float(r[spec['denominator']]) for r in rows)
        else:
            raise ValueError(f'Unknown aggregation: {op}')
    return format(value * spec.get('scale', 1), spec.get('format', '.4f'))


def normalize(text):
    text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)
    return re.sub(r'\s+', ' ', text.replace('\u2212', '-').replace('\u2013', '-').replace('\u00ad', ''))


def check_claims(text, claims, root=ROOT):
    text = normalize(text)
    results = []
    for claim in claims:
        if claim['label'] not in LABELS or claim['label'] == 'Held-out test':
            raise ValueError('Numeric claims must have a development evidence label')
        region = text
        if 'start' in claim:
            if claim['start'] not in region or claim['end'] not in region.split(claim['start'], 1)[1]:
                region = ''
            else:
                region = region.split(claim['start'], 1)[1].split(claim['end'], 1)[0]
        matches = list(re.finditer(claim['pattern'], region))
        if len(matches) != 1:
            results.append(dict(claim=claim['id'], label=claim['label'], value='', expected='', actual='', status='missing_claim' if not matches else 'ambiguous_claim'))
            continue
        for i, spec in enumerate(claim['values'], 1):
            expected = expected_value(spec, root)
            for match in matches:
                actual = match.group(f'v{i}').replace(',', '')
                results.append(dict(claim=claim['id'], label=claim['label'], value=str(i),
                                    expected=expected, actual=actual,
                                    status='pass' if actual == expected.replace(',', '') else 'mismatch'))
    return results


def check_blank_cells(tex, text):
    # Check both source structure and extracted rendered table. Never inspect test data.
    integrated = 'Model & Checkpoint & ROC-AUC & AP & ROC-AUC & AP' in tex
    source_header, rendered_header, end = (
        ('Model & Checkpoint & ROC-AUC & AP & ROC-AUC & AP',
         'Model Checkpoint ROC-AUC AP ROC-AUC AP', 'Full-development validation values')
        if integrated else ('Model & Checkpoint & Val ROC-AUC', 'Test PR-AUC', 'Validation cells')
    )
    if source_header not in tex or rendered_header not in normalize(text):
        return False
    table = tex.split(source_header, 1)[1].split(r'\bottomrule', 1)[0]
    rows = [line.strip().rstrip('\\').split('&') for line in table.splitlines() if '&' in line]
    if len(rows) != 20 or any(len(row) != 6 or any(c.strip() for c in row[4:]) for row in rows):
        return False
    models = ['Majority', 'G0 decision tree' if integrated else 'Decision Tree',
              'Random Forest', 'G0 logistic (no PCA)' if integrated else 'Logistic (no PCA)',
              'PCA + logistic', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
    if {(r[0].strip(), r[1].strip()) for r in rows} != {(m, h) for m in models for h in ('6h', '1h')}:
        return False
    expected = ' '.join(' '.join(c.strip() for c in row[:4]) for row in rows)
    expected = expected.replace(r'\(', '').replace(r'\)', '').replace(r'\pm', ' ± ')
    after_header = normalize(text).split(rendered_header, 1)[1]
    if end not in after_header:
        return False
    rendered = after_header.split(end, 1)[0].strip()
    # PDF extraction sometimes inserts a space before a decimal point in math mode.
    rendered = re.sub(r'(?<=\d) +(?=\.\d)', '', rendered)
    return normalize(expected) == rendered


def check_selected_placeholders(tex, text):
    if 'Selected owner & Checkpoint & Brier' not in tex or 'Selected owner Checkpoint Brier Skill ECE Bias' not in normalize(text):
        return False
    table = tex.split('Selected owner & Checkpoint & Brier', 1)[1].split('\\bottomrule', 1)[0]
    rows = [line for line in table.splitlines() if line.startswith(('P4 &', 'P5 &'))]
    if len(rows) != 4 or any(len(line.split('&')) != 6 or
                            any(cell.strip().rstrip('\\').strip() for cell in line.split('&')[2:])
                            for line in rows):
        return False
    rendered = normalize(text).split('Selected owner Checkpoint Brier Skill ECE Bias', 1)[1].split('Pending artifacts:', 1)[0]
    return bool(re.fullmatch(r' P4 6h P4 1h P5 6h P5 1h ', rendered))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, default=ROOT / 'output/pdf/report.pdf')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'reports/report_claims.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/checks')
    args = parser.parse_args(argv)
    from pypdf import PdfReader
    pages = []
    for number, page in enumerate(PdfReader(args.pdf).pages, 1):
        lines = (page.extract_text() or '').splitlines()
        if lines and lines[-1].strip() == str(number):
            lines.pop()
        pages.append('\n'.join(line.rstrip() for line in lines))
    text = '\n'.join(pages)
    config = json.loads(args.manifest.read_text())
    results = check_claims(text, config['claims'])
    failures = [r for r in results if r['status'] != 'pass']
    tex = (ROOT / 'reports/report.tex').read_text()
    blank = check_blank_cells(tex, text)
    selected_blank = check_selected_placeholders(tex, text)
    labels = all(label in normalize(text) for label in LABELS)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'report_text.txt').write_text(text)
    with (args.output / 'report_numeric_diff.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, lineterminator='\n', fieldnames=['claim', 'label', 'value', 'expected', 'actual', 'status'])
        writer.writeheader()
        writer.writerows(results)
    paths = sorted({s['csv'] for c in config['claims'] for s in c['values']})
    summary = dict(passed=not failures and blank and selected_blank and labels, registered_claims=len(config['claims']),
                   checked_values=len(results), numeric_failures=len(failures), held_out_cells_blank=blank,
                   selected_calibration_cells_blank=selected_blank,
                   evidence_labels_present=labels, scope=config['scope'],
                   pdf_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                   manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                   csv_sha256={p: hashlib.sha256(artifact_path(ROOT, p).read_bytes()).hexdigest() for p in paths})
    (args.output / 'report_check.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
