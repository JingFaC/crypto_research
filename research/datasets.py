"""Resolve preserved data versions through the Markdown provenance manifest."""
from pathlib import Path
import hashlib
import numpy as np


class DataStore:
    def __init__(self, data_root, legacy_data_root=None):
        self.root = Path(data_root).resolve()
        self.legacy = Path(legacy_data_root).resolve() if legacy_data_root else None
        self.index = {}
        manifest = self.root.parent / 'docs' / '数据' / '数据清单.md'
        if manifest.exists():
            for line in manifest.read_text(encoding='utf-8-sig').splitlines()[4:]:
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                if len(cells) == 8:
                    self.index[(cells[0].replace('\\', '/'), cells[3])] = (cells[1], cells[7])

    def resolve(self, relative, kind='原文件'):
        key = (relative.replace('\\', '/'), kind)
        if key in self.index:
            rel, digest = self.index[key]
            path = (self.root.parent / rel).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Manifest points outside data root')
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Input SHA256 mismatch: {path}')
            return path
        if self.legacy:
            rel = relative.removeprefix('backtest/data/')
            path = (self.legacy / rel).resolve()
            if not path.is_relative_to(self.legacy):
                raise ValueError('Input path outside legacy data root')
            if path.is_file():
                return path
        raise FileNotFoundError(f'Missing frozen input: {relative} ({kind})')

    def array(self, relative, kind='bars'):
        path = self.resolve(relative, kind if relative.endswith('.npz') else '原文件')
        array = np.load(path, allow_pickle=False)
        if path.suffix == '.npz':
            with array:
                return array[kind].copy()
        return array
