#!/usr/bin/env python3
"""
Verification script to check binary classification implementation
"""

import os
import re

def check_file_for_sedang(filepath):
    """Check if file contains 'sedang' references"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find all lines containing 'sedang'
        lines = content.split('\n')
        sedang_lines = []
        
        for i, line in enumerate(lines, 1):
            if 'sedang' in line.lower():
                sedang_lines.append((i, line.strip()))
        
        return sedang_lines
    except Exception as e:
        return [f"Error reading file: {e}"]

def main():
    print("🔍 CHECKING BINARY CLASSIFICATION IMPLEMENTATION")
    print("=" * 60)
    print("Searching for remaining 'sedang' references...")
    print()
    
    # Files to check
    files_to_check = [
        'setup_database_tables.py',
        'simple_sensor_dashboard.py', 
        'ml_dashboard.py',
        'ml_engine.py'
    ]
    
    total_issues = 0
    
    for filename in files_to_check:
        if os.path.exists(filename):
            print(f"📄 Checking {filename}...")
            sedang_lines = check_file_for_sedang(filename)
            
            if sedang_lines:
                if isinstance(sedang_lines[0], tuple):
                    print(f"  ❌ Found {len(sedang_lines)} 'sedang' references:")
                    for line_num, line_content in sedang_lines:
                        print(f"    Line {line_num}: {line_content}")
                    total_issues += len(sedang_lines)
                else:
                    print(f"  ❌ {sedang_lines[0]}")
                    total_issues += 1
            else:
                print(f"  ✅ No 'sedang' references found")
        else:
            print(f"  ⚠️ File {filename} not found")
        print()
    
    print("=" * 60)
    if total_issues == 0:
        print("🎉 BINARY CLASSIFICATION COMPLETE!")
        print("✅ All 'sedang' references have been removed")
        print("✅ System now uses only 'baik' and 'buruk'")
    else:
        print(f"⚠️ Found {total_issues} remaining 'sedang' references")
        print("💡 These need to be updated to binary classification")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
