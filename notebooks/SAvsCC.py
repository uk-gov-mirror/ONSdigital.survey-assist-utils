# %%
import pandas as pd

# %%
file = pd.read_parquet("../review_app/two_prompt_outputs/STG5_unambig90.parquet")

# %%
file['All_Clerical_codes'][0]

# %%
for i in range(len(file)):
    if ", nan, nan]" not in file['All_Clerical_codes'][i]:
        print(file['All_Clerical_codes'][i])

# %%
# Create a new column comparing initial and clerically codded matches
file['initial_and_clerical_match'] = None
for i in range(len(file)):
    if file['initial_code'][i] not in file['All_Clerical_codes'][i]:
        file['initial_and_clerical_match'][i] = False
    else:
        file['initial_and_clerical_match'][i] = True

# %%
# count the difference
count = 90
for i in range(len(file)):
    count -= (file['initial_code'][i] in file['All_Clerical_codes'][i])
print(count)

# %%
# count the number of rows, where SA could not assign a code unabmiguously
count = 0
for i in range(len(file)):
  if not file['unambiguously_codable'][i]:
    count += 1
    print(i, ":\nCC: ", file ['All_Clerical_codes'][i], "\nSA: ", file['initial_code'][i])
print(count)

# %%
file[['unambiguously_codable', 'unambiguously_codable_final']]

# %%
# file.to_parquet("STG5_unambig90_initial_and_clerical_match.parquet", index=False)


