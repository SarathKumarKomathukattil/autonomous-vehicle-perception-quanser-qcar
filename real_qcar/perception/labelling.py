import os

labels_path = r'C:\Users\kcksa\Documents\Quanser\5_research\pal_utilities\training_data\labels.txt'

with open(labels_path, 'w', encoding='utf-8') as f:
    f.write('yellow_line\n')
    f.write('roundabout\n')

print("labels.txt updated!")
print("Labels: yellow_line, roundabout")