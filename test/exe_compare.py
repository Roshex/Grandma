import sys
import os
import json
import argparse
import difflib
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Compare output directories of two tool versions.")
    parser.add_argument("-old", help="Path to the old tool's output directory")
    parser.add_argument("-new", help="Path to the new tool's output directory")
    parser.add_argument("-log", help="Path to write the comparison log")
    parser.add_argument("-ignore-prefix", action="store_true", default=True, 
                        help="Attempt to auto-detect and ignore file prefixes (default: True)")
    return parser.parse_args()

def deep_sort(obj):
    """
    Recursively sort JSON-like structures (dicts and lists) to ensure deterministic ordering.
    """
    if isinstance(obj, dict):
        return {k: deep_sort(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        # Sort list elements. We use str(x) as the key because dicts are not comparable in Python 3.
        return sorted([deep_sort(x) for x in obj], key=lambda x: str(x))
    return obj

def compare_text_content(lines1, lines2, filename1, filename2):
    """
    Compares two lists of strings and returns a formatted diff string.
    Skips lines that are identical (context=0).
    """
    diff = difflib.unified_diff(
        lines1, lines2,
        fromfile=f"OLD/{filename1}",
        tofile=f"NEW/{filename2}",
        n=0,  # No context lines, only changes
        lineterm=""
    )
    
    diff_lines = list(diff)
    if not diff_lines:
        return None
    
    return "\n".join(diff_lines)

def compare_json_files(path1, path2, fname1, fname2):
    try:
        with open(path1, 'r') as f1, open(path2, 'r') as f2:
            j1 = json.load(f1)
            j2 = json.load(f2)
        
        j1_sorted = deep_sort(j1)
        j2_sorted = deep_sort(j2)
        
        s1 = json.dumps(j1_sorted, indent=4, sort_keys=True).splitlines()
        s2 = json.dumps(j2_sorted, indent=4, sort_keys=True).splitlines()
        
        return compare_text_content(s1, s2, fname1, fname2)
    except Exception as e:
        return f"Error comparing JSON files: {str(e)}"

def compare_txt_files(path1, path2, fname1, fname2):
    try:
        with open(path1, 'r') as f1, open(path2, 'r') as f2:
            lines1 = [l.rstrip() for l in f1.readlines()]
            lines2 = [l.rstrip() for l in f2.readlines()]
        
        return compare_text_content(lines1, lines2, fname1, fname2)
    except Exception as e:
        return f"Error comparing text files: {str(e)}"

def get_file_map(directory, auto_detect_prefix=True):
    """
    Returns a dict mapping {normalized_name: full_filename}.
    Normalized name has the common prefix stripped.
    """
    path = Path(directory)
    files = sorted([f.name for f in path.iterdir() if f.is_file()])
    
    if not files:
        return {}, ""

    prefix = ""
    if auto_detect_prefix:
        prefix = os.path.commonprefix(files)
        # Only use prefix if it looks like a naming convention (e.g. ends in letter or - or .)
        if len(files) > 1 and len(prefix) > 2: 
            pass # Use found prefix
        else:
            prefix = "" # Unsafe prefix detection, fallback to full names

    file_map = {}
    for f in files:
        # Strip prefix for the key, keep full name as value
        key = f[len(prefix):] if prefix else f
        file_map[key] = f
        
    return file_map, prefix

def main():
    args = parse_args()
    
    d1 = Path(args.old)
    d2 = Path(args.new)
    log_path = Path(args.log)

    txt_extensions = ['.log'] #['.txt', '.log', '.tre']
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not d1.exists() or not d2.exists():
        print("One or both input directories do not exist.")
        sys.exit(1)

    # 1. Map files by normalized suffix (stripping unique prefixes like 'grampa'/'grandma')
    map1, prefix1 = get_file_map(d1)
    map2, prefix2 = get_file_map(d2)
    
    keys1 = set(map1.keys())
    keys2 = set(map2.keys())
    
    common_keys = sorted(keys1.intersection(keys2))
    only_old = sorted(keys1 - keys2)
    only_new = sorted(keys2 - keys1)
    
    with open(log_path, 'w') as log:
        log.write(f"COMPARISON REPORT\n")
        log.write(f"Old Dir: {d1} (Detected Prefix: '{prefix1}')\n")
        log.write(f"New Dir: {d2} (Detected Prefix: '{prefix2}')\n")
        log.write("=" * 60 + "\n\n")
        
        # Report Structure Diffs
        if only_old:
            log.write("FILES ONLY IN OLD DIR:\n")
            for k in only_old:
                log.write(f"  - {map1[k]}\n")
            log.write("\n")
            
        if only_new:
            log.write("FILES ONLY IN NEW DIR:\n")
            for k in only_new:
                log.write(f"  + {map2[k]}\n")
            log.write("\n")
            
        log.write("-" * 60 + "\n\n")
        
        # Compare Content
        files_with_diffs = 0
        
        for key in common_keys:
            fname1 = map1[key]
            fname2 = map2[key]
            
            p1 = d1 / fname1
            p2 = d2 / fname2
            
            result = None
            
            # Decide comparison method based on extension
            if key.endswith('.json'):
                result = compare_json_files(p1, p2, fname1, fname2)
            elif any(key.endswith(ext) for ext in txt_extensions):
                result = compare_txt_files(p1, p2, fname1, fname2)
            else:
                log.write(f"SKIP: {fname1} vs {fname2} (Unsupported extension)\n")
                continue
                
            if result:
                files_with_diffs += 1
                log.write(f"DIFF: {fname1} vs {fname2}\n")
                log.write(result)
                log.write("\n" + "-" * 40 + "\n\n")

        if files_with_diffs == 0:
            log.write("\nNo content differences found in matching files.\n")
        else:
            log.write(f"\nFound differences in {files_with_diffs} matching files.\n")

    print(f"Comparison complete. Report written to {log_path}")

if __name__ == "__main__":
    main()