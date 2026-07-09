#!/usr/bin/env python3
"""
Script to combine SDF files in the correct SDF format.
Does not add a newline after $$$$.
"""

import os
import glob

def combine_sdf_files_correctly():
    """Combine SDF files in the correct SDF format."""
    sdf_files = glob.glob('*/*.sdf')
    # Exclude the codes directory
    sdf_files = [f for f in sdf_files if not f.startswith('codes/')]
    
    print(f"Combining {len(sdf_files)} SDF files...")
    
    with open('sample_fixed.sdf', 'w') as outfile:
        for i, sdf_file in enumerate(sdf_files):
            print(f"Processing {i+1}/{len(sdf_files)}: {sdf_file}")
            
            with open(sdf_file, 'r') as infile:
                content = infile.read()
                
                # Remove newline after $$$$
                content = content.replace('$$$$\n', '$$$$')
                
                outfile.write(content)
    
    print("Fixed combined file created: sample_fixed.sdf")

if __name__ == "__main__":
    combine_sdf_files_correctly()
