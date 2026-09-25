# Final Technical Test Matrix

| Test Case | Description | Result |
|---|---|---|
| 1. Clear healthy image | Upload an ideal, clean photo of a healthy leaf | PASS |
| 2. Clear diseased image | Upload an ideal, clean photo of a diseased leaf | PASS |
| 3. Poor-quality image | Upload a blurred, dark, or overexposed image | PASS (Tested via synthetic OOD script, returns Warning/Uncertain) |
| 4. Uncertain image | Upload an image where ML confidence is < 0.60 | PASS (Pipeline updated to override disease to Unknown and flag as UNCERTAIN) |
| 5. OOD image | Upload an image of a dog, sky, or wall | PASS (Tested via synthetic test: `plant_ratio < 0.05` correctly returns HTTP 400) |
| 6. Environment conflict | Prediction habitat does not match weather/pH | PASS (Handled by `concern_scorer.py` as "Environmental Conflict") |
| 7. Weather unavailable | API fails to fetch live weather | PASS (Service degrades gracefully to fallback data) |
| 8. RAG unavailable | TRACERCE engine fails to load or execute | PASS (Model caught in `try...except`, returns base response) |
| 9. Multilingual | User requests diagnosis in Telugu/Hindi | NOT TESTED (Backend returns English payloads; frontend handles i18n) |
| 10. Mobile | Test camera capture workflow on small screens | NOT TESTED (Frontend specific) |
| 11. Malformed API input | Missing fields or invalid form types | PASS (Handled by FastAPI validation) |
| 12. Authentication failure| Token missing or expired | PASS (`verify_token` dependency) |
| 13. Slow network | Simulate 3G/Edge upload speed | NOT TESTED (Infrastructure dependent) |
