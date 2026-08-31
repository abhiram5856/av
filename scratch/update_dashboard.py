import os

analytics_path = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\frontend\src\app\dashboard\analytics\page.tsx"
history_path = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\frontend\src\app\dashboard\history\page.tsx"

# 1. ANALYTICS
with open(analytics_path, "r", encoding="utf-8") as f:
    analytics = f.read()

analytics = analytics.replace("severity_level", "concern_level")
analytics = analytics.replace("severity_score", "concern_score")
analytics = analytics.replace("totalSeverity", "totalConcern")
analytics = analytics.replace("avgSeverity", "avgConcern")
analytics = analytics.replace("Severity over time", "Concern Score over time")
analytics = analytics.replace("severity score progression", "concern score progression")
analytics = analytics.replace("Critical", "Critical Attention Required")
analytics = analytics.replace("High", "High Concern")
analytics = analytics.replace("Medium", "Moderate Concern")
analytics = analytics.replace("Healthy", "Low Concern")

with open(analytics_path, "w", encoding="utf-8") as f:
    f.write(analytics)

# 2. HISTORY
with open(history_path, "r", encoding="utf-8") as f:
    history = f.read()

history = history.replace("severity_level", "concern_level")
history = history.replace("getSeverityColor", "getConcernColor")
history = history.replace("Severity", "Concern Level")
history = history.replace("Critical", "Critical Attention Required")
history = history.replace("High", "High Concern")
history = history.replace("Medium", "Moderate Concern")
history = history.replace("Healthy", "Low Concern")

with open(history_path, "w", encoding="utf-8") as f:
    f.write(history)

print("Analytics and History updated")
