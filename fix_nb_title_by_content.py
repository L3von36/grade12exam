import json
from pathlib import Path
nb_path = Path(r"c:\Users\leo\Documents\grade 12\ethiopian_exam_pattern.ipynb")
nb = json.loads(nb_path.read_text(encoding='utf-8'))
restored = [
    "# Ethiopian Grade 12 Exam Pattern Project (Google Colab)",
    "",
    "This notebook converts your scripts into a Colab workflow:",
    "1. Download Grade 12 exam PDFs",
    "2. OCR text extraction from PDFs",
    "3. Build a simple question-pattern analysis baseline",
    ""
]
changed = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'markdown':
        src = ''.join(cell.get('source') or [])
        if 'Ethiopian Grade 12 Exam Pattern' in src:
            cell['source'] = restored
            changed = True
            print('Patched cell')
            break
if changed:
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
else:
    print('Title markdown not found')
