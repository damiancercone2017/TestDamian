import pandas as pd
import random

# Parameters for sample data
num_providers = 10
num_departments = 5
num_records = 100

# Sample providers and departments
providers = [f'Provider_{i+1}' for i in range(num_providers)]
departments = [f'Department_{i+1}' for i in range(num_departments)]

# Generate sample data
sample_data = []
for _ in range(num_records):
    provider = random.choice(providers)
    department = random.choice(departments)
    date = pd.Timestamp('2026-02-01') + pd.to_timedelta(random.randint(0, 29), unit='D')
    wrvu = random.uniform(0, 100)
    revenue = wrvu * random.uniform(100, 500)  # Random revenue based on wRVU
    additional_metric = random.uniform(0, 1)  # Just an example of an additional metric
    sample_data.append([provider, department, date, wrvu, revenue, additional_metric])

# Create DataFrame
columns = ['Provider', 'Department', 'Date', 'wRVU', 'Revenue', 'Additional Metric']
df = pd.DataFrame(sample_data, columns=columns)

# Write to Excel file
output_file = 'sample_data.xlsx'
with pd.ExcelWriter(output_file) as writer:
    df.to_excel(writer, sheet_name='wRVU_data', index=False)

print(f'Sample data generated and saved to {output_file}')
