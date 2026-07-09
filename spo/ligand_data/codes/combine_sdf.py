#!/usr/bin/env python3
"""
Script to combine SDF files with metadata into a single file.
"""

import os
import glob

def combine_sdf_files():
    """Combine all SDF files into one."""
    sdf_files = glob.glob('*/*.sdf')
    
    print(f"Combining {len(sdf_files)} SDF files...")
    
    with open('sample_with_metadata.sdf', 'w') as outfile:
        for i, sdf_file in enumerate(sdf_files):
            print(f"Processing {i+1}/{len(sdf_files)}: {sdf_file}")
            
            with open(sdf_file, 'r') as infile:
                content = infile.read()
                outfile.write(content)
    
    print("Combined file created: sample_with_metadata.sdf")

if __name__ == "__main__":
    combine_sdf_files()
