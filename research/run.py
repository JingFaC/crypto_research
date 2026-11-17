"""Unified offline research entry point: python -m research.run GROUP [options]."""
import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=['soxl', 'dca_macd', 'st_boll'])
    args, remaining = parser.parse_known_args()
    module = importlib.import_module(f'research.{args.group}.run')
    module.main(remaining)


if __name__ == '__main__':
    main()
