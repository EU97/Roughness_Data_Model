"""Organize.py — Roughness Data Organizer

Scans a source directory for .tx1/.tx2/.tx3 measurement triplets from Surfcom
profilometers and organizes them into the standard Grupo/Espe directory
structure used by Single.py and Batch.py.

Usage:
    python src/Organize.py <source_dir> [<dest_dir>] [options]
"""

import os
import sys
import shutil
import argparse
from collections import defaultdict

ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
         'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX','XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX']


def to_roman(n):
    """Convert 1-based integer to Roman numeral string (up to XX)."""
    if 1 <= n <= len(ROMAN):
        return ROMAN[n - 1]
    return str(n)


def find_tx_triplets(source_dir):
    """Scan *source_dir* recursively for complete .tx1/.tx2/.tx3 triplets.

    Returns a sorted list of dicts::

        {'stem': str, 'dir': str,
         'files': {'.tx1': path, '.tx2': path, '.tx3': path}}
    """
    tx_files = defaultdict(dict)  # (dir, stem) -> {ext: full_path}

    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            lower = fname.lower()
            for ext in ('.tx1', '.tx2', '.tx3'):
                if lower.endswith(ext):
                    stem = fname[:len(fname) - len(ext)]
                    full_path = os.path.join(root, fname)
                    tx_files[(root, stem)][ext] = full_path
                    break

    triplets = []
    for (directory, stem), ext_map in sorted(tx_files.items()):
        if '.tx1' in ext_map and '.tx2' in ext_map and '.tx3' in ext_map:
            triplets.append({
                'stem': stem,
                'dir': directory,
                'files': ext_map,
            })
        else:
            missing = [e for e in ('.tx1', '.tx2', '.tx3') if e not in ext_map]
            print(f"  [WARN] Incomplete triplet '{stem}' in {directory} "
                  f"— missing: {', '.join(missing)}")

    return triplets


def organize(source_dir, dest_dir, *, move=False, dry_run=False,
             specimens_per_group=5, single_group=False, prefix='Grupo'):
    """Organize triplets into ``<prefix><Roman>/<Espe><Roman>`` structure."""
    triplets = find_tx_triplets(source_dir)

    if not triplets:
        print("No complete .tx1/.tx2/.tx3 triplets found.")
        return

    print(f"Found {len(triplets)} specimen triplet(s).\n")

    # Build groups
    if single_group:
        groups = [triplets]
    else:
        n = max(1, specimens_per_group)
        groups = [triplets[i:i + n] for i in range(0, len(triplets), n)]

    verb = "Move" if move else "Copy"
    if dry_run:
        verb = f"[DRY-RUN] Would {verb.lower()}"

    for g_idx, group in enumerate(groups, start=1):
        group_name = f"{prefix}{to_roman(g_idx)}"
        for s_idx, triplet in enumerate(group, start=1):
            espe_name = f"Espe{to_roman(s_idx)}"
            dest_folder = os.path.join(dest_dir, group_name, espe_name)

            if not dry_run:
                os.makedirs(dest_folder, exist_ok=True)

            for ext in ('.tx1', '.tx2', '.tx3'):
                src_path = triplet['files'][ext]
                dst_path = os.path.join(dest_folder, os.path.basename(src_path))
                print(f"  {verb}: {src_path} -> {dst_path}")
                if not dry_run:
                    if move:
                        shutil.move(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)

    action_word = 'Would organize' if dry_run else 'Organized'
    print(f"\n{action_word} {len(triplets)} specimen(s) "
          f"into {len(groups)} group(s).")


def main():
    parser = argparse.ArgumentParser(
        description='Organize .tx1/.tx2/.tx3 triplets into Grupo/Espe structure.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/Organize.py raw_data/ data/
  python src/Organize.py raw_data/ data/ --move
  python src/Organize.py raw_data/ data/ --dry-run
  python src/Organize.py raw_data/ data/ --specimens-per-group 3
  python src/Organize.py raw_data/ data/ --single-group
""")
    parser.add_argument('source',
                        help='Source directory containing .tx1/.tx2/.tx3 files.')
    parser.add_argument('dest', nargs='?', default='data',
                        help='Destination root directory (default: data).')
    parser.add_argument('--move', action='store_true',
                        help='Move files instead of copying (default: copy).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show actions without executing them.')
    parser.add_argument('--specimens-per-group', type=int, default=5,
                        help='Number of specimens per group (default: 5).')
    parser.add_argument('--single-group', action='store_true',
                        help='Put all specimens in a single group.')
    parser.add_argument('--prefix', default='Grupo',
                        help="Group folder prefix (default: 'Grupo').")

    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"Error: Source directory '{args.source}' not found.")
        sys.exit(1)

    print(f"Scanning: {os.path.abspath(args.source)}")
    print(f"Destination: {os.path.abspath(args.dest)}")
    print(f"Mode: {'Move' if args.move else 'Copy'}")
    if args.dry_run:
        print("** DRY RUN — no files will be modified **")
    print()

    organize(
        source_dir=args.source,
        dest_dir=args.dest,
        move=args.move,
        dry_run=args.dry_run,
        specimens_per_group=args.specimens_per_group,
        single_group=args.single_group,
        prefix=args.prefix,
    )


if __name__ == '__main__':
    main()
