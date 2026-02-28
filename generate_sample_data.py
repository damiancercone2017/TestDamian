import random
import pandas as pd

# Configuration
CLINICS = [f"CLINIC_{i + 1:02d}" for i in range(8)]
ROWS_PER_CLINIC = 5  # multiple rows per clinic to exercise groupby summing

# Generate sample data
sample_data = []
for clinic in CLINICS:
    for _ in range(ROWS_PER_CLINIC):
        wrvu = round(random.uniform(50, 300), 2)
        revenue = round(wrvu * random.uniform(80, 200), 2)
        sample_data.append({"clinic_code": clinic, "revenue": revenue, "wRVU": wrvu})

df = pd.DataFrame(sample_data)

# Primary output: CSV compatible with app.py
csv_file = "sample_clinic_data.csv"
df.to_csv(csv_file, index=False)
print(f"Sample data generated and saved to {csv_file}")

# Optional: Excel output
# xlsx_file = "sample_clinic_data.xlsx"
# df.to_excel(xlsx_file, index=False)
# print(f"Also saved to {xlsx_file}")
