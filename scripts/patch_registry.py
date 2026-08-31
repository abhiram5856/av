import re
from pathlib import Path

path = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\models\class_registry.py")

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Extract the current CLASS_NAMES
match = re.search(r"CLASS_NAMES: List\[str\] = \[(.*?)\]", content, re.DOTALL)
if match:
    classes = [c.strip().strip('"').strip("'") for c in match.group(1).split(",") if c.strip()]
else:
    print("Could not find CLASS_NAMES")
    exit(1)

display_dict = []
disease_dict = []
crop_dict = []

for c in classes:
    # Display name
    display_name = c.replace("_", " ").title()
    display_dict.append(f'    "{c}": "{display_name}",')
    
    # Disease flag
    is_disease = "False" if "healthy" in c else "True"
    disease_dict.append(f'    "{c}": {is_disease},')
    
    # Crop family
    crop = c.split("_")[0]
    crop_dict.append(f'    "{c}": "{crop}",')

# Build the replacements
disp_str = "DISPLAY_NAMES: Dict[str, str] = {\n" + "\n".join(display_dict) + "\n}"
dis_str = "IS_DISEASE: Dict[str, bool] = {\n" + "\n".join(disease_dict) + "\n}"
crop_str = "CROP_FAMILY: Dict[str, str] = {\n" + "\n".join(crop_dict) + "\n}"

content = re.sub(r"DISPLAY_NAMES: Dict\[str, str\] = \{.*?\}", disp_str, content, flags=re.DOTALL)
content = re.sub(r"IS_DISEASE: Dict\[str, bool\] = \{.*?\}", dis_str, content, flags=re.DOTALL)
content = re.sub(r"CROP_FAMILY: Dict\[str, str\] = \{.*?\}", crop_str, content, flags=re.DOTALL)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Registry patched successfully!")
