import json, sys, pathlib
path = r"c:/Users/leo/Documents/grade 12/ethiopian_exam_pattern_new.ipynb"
try:
    json.load(open(path, encoding='utf-8'))
    print('JSON parsed successfully')
except Exception as e:
    print('Error parsing JSON:', e)
