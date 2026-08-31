import sys
print("Starting import...")
try:
    import backend.main
    print("Import successful!")
except Exception as e:
    print("Error:", e)
