#!/usr/bin/env python3
"""
Diagnostic script to check available ETR (External Trigger Range) constants
in the AlazarTech API for your specific board.
"""

import sys

try:
    sys.path.append('/usr/local/AlazarTech/samples/Samples_Python/Library/')
    import atsapi as ats
    print("✓ atsapi loaded successfully\n")
    
    # Find all ETR constants
    etr_constants = [attr for attr in dir(ats) if attr.startswith('ETR_')]
    
    if etr_constants:
        print(f"Found {len(etr_constants)} ETR constants:")
        print("-" * 50)
        for const in sorted(etr_constants):
            try:
                value = getattr(ats, const)
                print(f"  {const:20s} = {value}")
            except:
                print(f"  {const:20s} = <error reading value>")
    else:
        print("⚠ No ETR_ constants found in atsapi")
        print("\nSearching for similar constants...")
        trigger_related = [attr for attr in dir(ats) if 'TRIG' in attr or 'EXT' in attr]
        print(f"\nFound {len(trigger_related)} trigger-related constants:")
        for const in sorted(trigger_related)[:30]:
            print(f"  - {const}")
    
    print("\n" + "=" * 50)
    print("RECOMMENDED FIX:")
    print("=" * 50)
    
    if 'ETR_2V5' in etr_constants:
        print("✓ Use 'ETR_2V5' for 2.5V external trigger range")
    elif 'ETR_2V' in etr_constants:
        print("✓ Use 'ETR_2V' for 2V external trigger range")
    elif 'ETR_5V' in etr_constants:
        print("⚠ ETR_2V5 not found. Use 'ETR_5V' or 'ETR_1V' instead")
    else:
        print("⚠ No standard ETR constants found")
        print("Check your ATS SDK version and board model support")
    
except ImportError as e:
    print(f"✗ Failed to import atsapi: {e}")
    print("\nPossible reasons:")
    print("  1. AlazarTech SDK not installed")
    print("  2. Path incorrect: /usr/local/AlazarTech/samples/Samples_Python/Library/")
    print("  3. Python version incompatible with SDK")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
