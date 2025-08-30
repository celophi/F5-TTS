#!/usr/bin/env python3
"""
validate_phoneme_list.py

Robust analysis of htsvoice files (used in openjtalk) focusing on actual phoneme patterns.
This can be used on mei_normal.htsvoice or nitech_jp_atr503_m001.htsvoice to try to figure out what possible Japanese phonemes there are.
The output can give some insight on building an approriate vocab.txt
"""

import re
import sys
from pathlib import Path

def analyze_voice_file(voice_path):
    """Analyze the voice file with multiple approaches to find real phonemes."""
    try:
        with open(voice_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        print(f"ERROR: {e}")
        return set()
    
    # Approach 1: Text decoding (may have false positives)
    try:
        text_content = data.decode('ascii', errors='ignore')
        print(f"File size: {len(data)} bytes")
        print(f"Decoded text length: {len(text_content)} characters")
    except:
        print("Failed to decode as text")
        return set()
    
    # Approach 2: Look for actual phoneme patterns more carefully
    print("\n=== PATTERN ANALYSIS ===")
    
    # Look for well-formed phoneme patterns (more specific regex)
    well_formed_patterns = set()
    pattern_matches = re.findall(r'[^a-z]\{([a-z]+(?:,[a-z]+)*)\}[^a-z]', text_content)
    for match in pattern_matches:
        patterns = [p.strip() for p in match.split(',')]
        well_formed_patterns.update(patterns)
        print(f"Found pattern: {{{match}}}")
    
    # Look for individual phonemes in likely contexts
    individual_phonemes = set()
    phoneme_matches = re.findall(r'[\-\+]([a-z]{1,2})[\-\+]', text_content)
    individual_phonemes.update(phoneme_matches)
    
    # Look for phonemes in question contexts
    question_phonemes = set()
    question_matches = re.findall(r'[\/\"]([a-z]{1,2})[\/\"]', text_content)
    question_phonemes.update(question_matches)
    
    print(f"Well-formed patterns: {sorted(well_formed_patterns)}")
    print(f"Individual phonemes: {sorted(individual_phonemes)}")
    print(f"Question phonemes: {sorted(question_phonemes)}")
    
    # Combine and filter for realistic Japanese phonemes
    all_found = well_formed_patterns | individual_phonemes | question_phonemes
    
    # Standard Japanese phoneme set
    japanese_phonemes = {
        'a', 'i', 'u', 'e', 'o', 'b', 'by', 'ch', 'cl', 'd', 'f', 'g', 'gy',
        'h', 'hy', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'N', 'p', 'py', 'r',
        'ry', 's', 'sh', 't', 'ts', 'w', 'y', 'z', 'pau', 'sil'
    }
    
    realistic_phonemes = all_found & japanese_phonemes
    questionable = all_found - japanese_phonemes
    
    print(f"\nRealistic Japanese phonemes found: {sorted(realistic_phonemes)}")
    print(f"Questionable patterns (likely false positives): {sorted(questionable)}")
    
    return realistic_phonemes

def hex_dump_suspicious_bytes(voice_path, search_patterns):
    """Hex dump around areas with suspicious patterns."""
    try:
        with open(voice_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    print(f"\n=== HEX ANALYSIS OF SUSPICIOUS PATTERNS ===")
    
    for pattern in search_patterns:
        pattern_bytes = pattern.encode('ascii')
        position = data.find(pattern_bytes)
        
        if position != -1:
            print(f"\nFound '{pattern}' at position 0x{position:08x}")
            start = max(0, position - 16)
            end = min(len(data), position + len(pattern_bytes) + 16)
            
            print("Context bytes:")
            for i in range(start, end, 16):
                line_bytes = data[i:min(i+16, end)]
                hex_str = ' '.join(f'{b:02x}' for b in line_bytes)
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in line_bytes)
                print(f"{i:08x}: {hex_str:<48} {ascii_str}")

def main():
    voice_path = "mei_normal.htsvoice"
    
    if not Path(voice_path).exists():
        print(f"Please place {voice_path} in the current directory")
        sys.exit(1)
    
    print(f"Analyzing: {voice_path}")
    
    # Analyze for real phonemes
    realistic_phonemes = analyze_voice_file(voice_path)
    
    # Hex dump around suspicious patterns to see their real context
    hex_dump_suspicious_bytes(voice_path, ['l', 'uz', 'xx', 'q'])
    
    # Final authoritative list (based on standard + what we found)
    authoritative_phonemes = {
        'a', 'i', 'u', 'e', 'o', 'b', 'by', 'ch', 'cl', 'd', 'f', 'g', 'gy',
        'h', 'hy', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'N', 'p', 'py', 'r',
        'ry', 's', 'sh', 't', 'ts', 'w', 'y', 'z', 'pau', 'sil'
    }
    
    # Add any realistic phonemes we found that are in the standard set
    authoritative_phonemes.update(realistic_phonemes)
    
    print(f"\n=== FINAL AUTHORITATIVE PHONEME SET ===")
    print(f"Total phonemes: {len(authoritative_phonemes)}")
    print("Sorted list:")
    for i, phoneme in enumerate(sorted(authoritative_phonemes), 1):
        print(f"  {i:2d}. {phoneme}")

if __name__ == "__main__":
    main()