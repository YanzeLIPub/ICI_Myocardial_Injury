"""Reject files outside the reviewed public-code inventory."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 250_000


def check_release():
    inventory = ROOT / 'PUBLIC_FILES'
    expected = inventory.read_text(encoding='utf-8').splitlines()
    if expected != sorted(set(expected)) or not expected:
        raise ValueError('PUBLIC_FILES must contain unique sorted paths.')
    for name in expected:
        path = ROOT / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid inventory path.')
        if path.is_symlink() or ROOT not in path.resolve().parents:
            raise ValueError('Linked or out-of-tree file: ' + name)
        payload = path.read_bytes()
        if len(payload) > MAX_TEXT_BYTES or b'\x00' in payload:
            raise ValueError('Large or binary file: ' + name)
        text = payload.decode('utf-8')
        patterns = [
            r'(?i)(?:gh[pousr]_[a-z0-9]{30,}|github_pat_[a-z0-9_]{30,})',
            r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
            r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s]+',
            r'(?<!\\)\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._-]+',
            r'/(?:home|Users|mnt|media|root|tmp|var/tmp)/[A-Za-z0-9._/-]+',
            r'[\u3400-\u9fff]',
        ]
        if any(re.search(pattern, text) for pattern in patterns):
            raise ValueError('Restricted content pattern in: ' + name)
    if (ROOT / '.git').exists():
        result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT,
                                check=True, capture_output=True)
        tracked = set(result.stdout.decode('utf-8').rstrip('\0').split('\0'))
        tracked.discard('')
        if tracked != set(expected):
            raise ValueError('Tracked files differ from PUBLIC_FILES.')
    print(f'Public-code inventory passed: {len(expected)} reviewed text files.')


if __name__ == '__main__':
    try:
        check_release()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
