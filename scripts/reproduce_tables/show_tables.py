"""Print existing confirmatory and historical tables; no statistical recomputation."""
import csv
from pathlib import Path
def main():
    root=Path(__file__).resolve().parents[2]
    for rel in ['results/confirmatory/ACROSS_SEED_SUMMARY.csv','results/confirmatory/PRIMARY_PAIRED_EFFECTS.csv','results/historical/additional_samples/NEW_VALIDATION_SUMMARY.csv']:
        print('\n'+rel)
        with (root/rel).open(encoding='utf-8-sig',newline='') as f:
            for row in csv.reader(f):print(' | '.join(row))
if __name__=='__main__':main()
