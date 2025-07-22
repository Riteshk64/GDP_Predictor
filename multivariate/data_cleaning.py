import pandas as pd

file_names = [
    "current_account_balance.xlsx",
    "exports.xlsx",
    "gdp.xlsx",
    "household_consumption.xlsx",
    "imports.xlsx",
    "industrial_production_proxy.xlsx",
    "inflation.xlsx",
    "investment.xlsx",
]

years = [str(y) for y in range(1991, 2025)]

merged_df = None

for i, file in enumerate(file_names):
    df = pd.read_excel(f'multivariate/data/raw/{file}', skiprows=4)
    df.columns = df.columns.astype(str).str.strip()

    # Remove quotes from all string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip('"').str.strip("'")

    df_india = df[df['Country Name'] == 'India']
    param_name = df_india['Indicator Name'].values[0]
    df_years = df_india[years].T.reset_index()
    df_years.columns = ['Year', param_name]

    # Scale GDP values to billions
    if 'gdp' in file:
        df_years[param_name] = pd.to_numeric(df_years[param_name], errors='coerce') / 1e9

    if merged_df is None:
        merged_df = df_years
    else:
        merged_df = pd.merge(merged_df, df_years, on='Year')

# Save the cleaned and merged data
merged_df.to_excel('multivariate/data/processed/gdp_factor_india.xlsx', index=False)
merged_df.to_csv('multivariate/data/processed/gdp_factor_india.csv', index=False)