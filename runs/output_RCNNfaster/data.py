import csv

# Lire sign_names.csv
sign_map = {}
with open(r"sign_names.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sign_map[row["SignName"]] = int(row["ClassId"])

# Puis dans ton loop pour les annotations :
category_id = sign_map[cls]  # cls = class_name de ton annotation