import os
import glob

def replace_in_file(path, old, new):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {path}")

# Path to the research directory
base_dir = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research"

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            replace_in_file(path, "SeverityContext", "ConcernContext")
            replace_in_file(path, "extract_severity_evidence", "extract_concern_evidence")
            replace_in_file(path, "final_severity_index", "concern_score")
            replace_in_file(path, "urgency_level", "concern_level")
            replace_in_file(path, "context.severity", "context.concern")

# Update test files too
test_dir = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\tests"
for root, _, files in os.walk(test_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            replace_in_file(path, "SeverityContext", "ConcernContext")
            replace_in_file(path, "extract_severity_evidence", "extract_concern_evidence")
            replace_in_file(path, "final_severity_index", "concern_score")
            replace_in_file(path, "urgency_level", "concern_level")
            replace_in_file(path, "context.severity", "context.concern")

print("Done")
