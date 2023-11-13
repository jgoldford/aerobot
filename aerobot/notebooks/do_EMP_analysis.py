import numpy as np
import pandas as pd

# Load the Earth microbiome project analysis and metadata
EMP_meta_df = pd.read_csv('../../results/EarthMicrobiome_metadata.tsv', sep='\t', index_col=0)
EMP_res_df = pd.read_csv('../../results/aa3mer.aerotype.EarthMicrobiome.csv', index_col=0)
EMP_merged = EMP_res_df.join(EMP_meta_df, how='outer')

# Earth Microbiome Project has a lot habitat name inconsistencies
# This is a semi-manual renaming of the habitats
# TODO: factor this out into CSV file
renaming_habitat = {
    'Human feces': 'Animal feces',
    'Human fecal': 'Animal feces',
    'Goate feces': 'Animal feces',
    'freshwater': 'Freshwater',
    'freshwater lake': 'Freshwater',
    'landfill leachate': 'Landfill leachate',
    'Pelagic marine': 'Marine',
    'Cattle and sheep rumen': 'Rumen',
    'rumen': 'Rumen',
    'hot spring sediment': 'Hot spring sediment',
    'wastewater': 'Wastewater',
    'hydrothermal vent': 'Hydrothermal vent',
    'sediment': 'Sediment',
    'activated sludge': 'Activated sludge',
    'Sheep rumen': 'Rumen',
    'Camel rumen': 'Rumen',
    'soil': 'Bulk soil',
    'compost': 'Compost',
    'defined medium': 'Defined medium',
    'Moose rumen': 'Rumen',
    'Bovine rumen': 'Rumen',
    'sewage': 'Sewage',
    'anaerobic enrichment culture': 'Anaerobic enrichment culture',
    'Activated Sludge': 'Activated sludge',
    'freshwater microbial mat': 'Freshwater microbial mat',
    'seawater': 'Seawater',
    'city subway': 'City subway',
    'City subway metal': 'City subway',
    'city subway metal': 'City subway',
    'city subway wood': 'City subway',
    'city subway metal/plastic': 'City subway',
    'Human host-associated': 'Human associated',
    'Human skin': 'Human associated',
    'saline water': 'Saline water',
    'Human': 'Human associated',
    'hydrothermal vent microbial mat': 'Hydrothermal vent microbial mat',
    'marine': 'Marine',
    'watersheds': 'Watersheds',
    'Fecal': 'Animal feces',
    'Capybara group fecal': 'Animal feces',
    'insecta': 'Insecta',
    'ant dump': 'Ant dump',
    'marine sediment': 'Marine sediment',
    'plant litter': 'Plant litter',
    'biosolids': 'Biosolids',
    'ant gut': 'Ant gut',
    'Human gut': 'Human associated',
    'Freshwater Sediment': 'Freshwater sediment',
    'Orangutan group fecal': 'Animal feces',
    'feces': 'Animal feces',
    'Elk feces': 'Animal feces',
    'corn rhizosphere': 'Rhizosphere',
    'city subway metalplastic': 'City subway',
    'Deep surbsurface': 'Deep subsurface',
    'deep subsurface': 'Deep subsurface',
    'fungus garden': 'Fungus garden',
    'surface seawater': 'Seawater',
    'lab-scale EBPR bioreactor': 'Bioreactor',
    'miscanthus rhizosphere': 'Rhizosphere',
    'Miscanthus rhizosphere': 'Rhizosphere',
    'Arabidopsis thaliana rhizosphere': 'Rhizosphere',
    'Arabidopsis rhizosphere': 'Rhizosphere',
    'Corn rhizosphere': 'Rhizosphere',
    'switchgrass rhizosphere': 'Rhizosphere',
    'Switchgrass rhizosphere': 'Rhizosphere',
    'rhizosphere': 'Rhizosphere',
    'Populus rhizosphere': 'Rhizosphere',
    'Corn, switchgrass and miscanthus rhizosphere': 'Rhizosphere',
    'Tabebuia heterophylla rhizosphere': 'Rhizosphere',
    'iron-sulfur acid spring': 'Iron-sulfur acid spring',
    'deep subsurface aquifer': 'Deep subsurface aquifer',
    'Rat Cecum': 'Rat cecum',
    'fungus gardens': 'Fungus garden',
    'bulk soil': 'Bulk soil',
    'leaf surface': 'Leaf surface',
    'Asian elephant fecal': 'Animal feces',
    'Eastern black-and-white colobus group fecal': 'Animal feces',
    'Western lowland gorilla individual fecal': 'Animal feces',
    'Ring-tailed lemur group fecal': 'Animal feces',
    'Orangutan individual fecal': 'Animal feces',
    'Huma fecal': 'Animal feces',
    'Lyns pardinus fecal': 'Animal feces',
    'Human oral': 'Human associated',
    'Human colon tissue': 'Human associated',
    'Human bile duct': 'Human associated',
    'Human lung': 'Human associated',
    'Premature human infant gut': 'Human associated',
    'anaerobic bioreactor biomass': 'Anaerobic bioreactor biomass',
    'food waste': 'Food waste',
    'sludge': 'Sludge',
    'raw primary sludge': 'Raw sludge',
    'active sludge': 'Activated sludge',
    'Active sludge': 'Activated sludge',
    'granular sludge': 'Granular sludge',
    'Anaerobic biogas reactor': 'Biogas fermentation',
}

# Make a processed habitat column that is more uniform
EMP_merged['habitat_processed'] = EMP_merged['habitat'].str.strip().replace(renaming_habitat)
print('Full dataset has', EMP_merged.shape[0], 'MAGs')
EMP_merged.to_csv('../../results/EMP_merged_raw.csv')

# Filter out low-completeness MAGs
mask = EMP_merged.completeness > 50
EMP_merged_subset = EMP_merged[mask]
print(EMP_merged_subset.shape[0], 'MAGs with >50% completeness')

# Filter out samples ("metagenomes") with less than 10 MAGs
genomes_per = EMP_merged_subset.groupby('metagenome_id').agg(dict(label='count'))
mask = genomes_per.label > 10
samples_w_enough = genomes_per[mask].index
print(samples_w_enough.size, 'samples with more than 10 MAGs')

mask2 = EMP_merged_subset.metagenome_id.isin(samples_w_enough) 
EMP_merged_subset = EMP_merged_subset[mask2]

# Filter our habitats with less than 10 samples
habitat_per = EMP_merged_subset.habitat_processed.value_counts()
mask3 = habitat_per > 10
habitat_w_enough = habitat_per[mask3].index
print(habitat_w_enough.size, 'remaining habitats with more than 10 samples')

mask4 = EMP_merged_subset.habitat_processed.isin(habitat_w_enough)
EMP_merged_subset = EMP_merged_subset[mask4]
print(EMP_merged_subset.shape[0], 'remaining MAGs')
print(EMP_merged_subset.habitat_processed.value_counts().head(10))

EMP_merged_subset.to_csv('../../results/EMP_merged_filtered.csv', index=False)

# Calculate the aerobe/anaerobe fractions for each sample
print('Calculating aerobe/anaerobe/facultative fractions for each sample')
ids = []
rows = []
for gid, gdf in EMP_merged_subset.groupby('metagenome_id'):
    counts = gdf.label.value_counts()
    normed_counts = counts.astype(float) / counts.sum()
    normed_counts.index = ['f_{0}'.format(c) for c in normed_counts.index]
    counts['total'] = counts.sum()
    rows.append(pd.concat([counts, normed_counts]))
    ids.append(gid)

fracs = pd.DataFrame(rows, index=ids)
fracs = fracs.fillna(0)

# Add the habitat information back in 
habitats_by_sample = EMP_merged.groupby('metagenome_id').habitat_processed.first()
fracs['habitat'] = habitats_by_sample.loc[fracs.index]
fracs['aerobe_anaerobe_ratio'] = fracs['f_Aerobe'] / fracs['f_Anaerobe']
fracs.to_csv('../../results/EMP_merged_filtered_fracs_by_sample.csv')

# Take the mean aerobe/anaerobe fractions across habitat categories
print('Calculating mean faction per habitat label')

agg_dict = {'Anaerobe': 'sum', 'Aerobe': 'sum', 'Facultative': 'sum', 'total': 'sum',
            'aerobe_anaerobe_ratio': np.nanmean}
mean_fracs = fracs.groupby('habitat').agg(agg_dict).sort_values('total', ascending=False)
mean_fracs['f_Anaerobe'] = mean_fracs['Anaerobe'] / mean_fracs['total']
mean_fracs['f_Aerobe'] = mean_fracs['Aerobe'] / mean_fracs['total']
mean_fracs['f_Facultative'] = mean_fracs['Facultative'] / mean_fracs['total']
mean_fracs.sort_values('f_Anaerobe', ascending=False)

mean_fracs['pct_Anaerobe'] = mean_fracs['f_Anaerobe'] * 100
mean_fracs['pct_Aerobe'] = mean_fracs['f_Aerobe'] * 100
mean_fracs['pct_Facultative'] = mean_fracs['f_Facultative'] * 100

mean_fracs.to_csv('../../results/EMP_merged_filtered_mean_fracs_by_habitat.csv')