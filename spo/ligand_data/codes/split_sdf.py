#!/usr/bin/env python3
"""
Script to split MOE-processed SDF files back to their original locations.
"""

import os
import re

def parse_sdf_with_metadata(sdf_file):
    """Parse an SDF file and extract metadata."""
    molecules = []
    
    with open(sdf_file, 'r') as f:
        content = f.read()
    
    # Split molecules by $$$$
    mol_blocks = content.split('$$$$')
    
    for mol_block in mol_blocks:
        if not mol_block.strip():
            continue
            
        # Extract metadata
        original_dir = None
        original_filename = None
        molecule_id = None
        
        lines = mol_block.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('> <Original_Directory>') or line.startswith('>  <Original_Directory>'):
                if i + 1 < len(lines):
                    original_dir = lines[i + 1].strip()
            elif line.startswith('> <Original_Filename>') or line.startswith('>  <Original_Filename>'):
                if i + 1 < len(lines):
                    original_filename = lines[i + 1].strip()
            elif line.startswith('> <Molecule_ID>') or line.startswith('>  <Molecule_ID>'):
                if i + 1 < len(lines):
                    molecule_id = lines[i + 1].strip()
        
        if original_dir and original_filename:
            molecules.append({
                'content': mol_block + '$$$$',
                'directory': original_dir,
                'filename': original_filename,
                'molecule_id': molecule_id
            })
    
    return molecules

def split_sdf_file(sdf_file):
    """Split an SDF file back to its original locations."""
    molecules = parse_sdf_with_metadata(sdf_file)
    
    print(f"Found {len(molecules)} molecules")
    
    for mol in molecules:
        # Create directory if it does not exist
        if not os.path.exists(mol['directory']):
            os.makedirs(mol['directory'])
        
        # File path
        file_path = os.path.join(mol['directory'], mol['filename'])
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(mol['content'])
        
        print(f"Saved: {file_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python split_sdf.py <sdf_file>")
        sys.exit(1)
    
    sdf_file = sys.argv[1]
    split_sdf_file(sdf_file)
