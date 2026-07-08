"""Fix unicode characters for Windows cp1252 compatibility."""
import os

files = [
    'utils.py', 'train_physio.py', 'train_facial.py',
    'train_fusion.py', 'models/fusion_model.py'
]

for f in files:
    path = os.path.join('D:/Capstone', f)
    with open(path, 'r', encoding='utf-8') as fh:
        content = fh.read()
    content = content.replace('\u2713', '[OK]')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f"Fixed: {f}")

print("Done!")
