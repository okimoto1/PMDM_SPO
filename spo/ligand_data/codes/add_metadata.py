#!/usr/bin/env python3
"""
Script to add metadata to SDF files:
- Original directory name
- Original filename
- Molecule ID
for each molecule.
"""

import os
import sys
import glob

def add_metadata_to_sdf(sdf_file, output_file):
    """Add metadata to an SDF file."""
    with open(sdf_file, 'r') as f:
        content = f.read()
    
    # Get filename and directory name
    dir_name = os.path.dirname(sdf_file)
    file_name = os.path.basename(sdf_file)
    
    # Extract molecule ID (e.g., gen_sample_10 -> sample_10)
    if 'gen_sample_' in file_name:
        molecule_id = file_name.split('gen_sample_')[1].replace('.sdf', '')
        molecule_id = f"sample_{molecule_id}"
    else:
        molecule_id = file_name.replace('.sdf', '')
    
    # Add metadata
    metadata = f"""
> <Original_Directory>
{dir_name}

> <Original_Filename>
{file_name}

> <Molecule_ID>
{molecule_id}

$$$$
"""
    
    # Insert metadata before the existing $$$$
    if '$$$$' in content:
        content = content.replace('$$$$', metadata)
    else:
        content += metadata
    
    with open(output_file, 'w') as f:
        f.write(content)

def process_all_sdf_files():
    """Process all SDF files."""
    sdf_files = glob.glob('*/*.sdf')
    
    print(f"Found {len(sdf_files)} SDF files")
    
    for sdf_file in sdf_files:
        # Temporary filename
        temp_file = sdf_file + '.temp'
        
        # Add metadata
        add_metadata_to_sdf(sdf_file, temp_file)
        
        # Replace the original file
        os.rename(temp_file, sdf_file)
        
        print(f"Processed: {sdf_file}")
    
    print("All files processed!")

if __name__ == "__main__":
    process_all_sdf_files()
