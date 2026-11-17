"""Local bundle publication and source identity; no model dependencies."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Publish one immutable JSON receipt without exposing a partial file."""
    path = Path(path)
    raw = json.dumps(value, indent=2, allow_nan=False) + '\n'
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.writing-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(raw)
        # Hard-link creation is atomic and refuses an existing destination.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def code_provenance(require_clean=True):
    # Caller cwd and inherited Git overrides must not select another repository.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}

    def git(*args):
        return subprocess.check_output(['git', '-C', str(SOURCE_ROOT), *args], env=env)

    commit = git('rev-parse', 'HEAD').decode().strip()
    diff = git('diff', '--binary', 'HEAD', '--')
    untracked = git('ls-files', '--others', '--exclude-standard', '-z', '--', 'src', 'scripts')
    digest = hashlib.sha256(diff)
    files = [p for p in untracked.decode().split('\0') if p.endswith('.py')]
    for name in sorted(files):
        digest.update(name.encode() + b'\0' + (SOURCE_ROOT / name).read_bytes())
    dirty = bool(diff or files)
    if dirty and require_clean:
        raise ValueError('source checkout is dirty; commit code first or explicitly allow dirty code')
    return {'commit': commit, 'dirty': dirty, 'changes_sha256': digest.hexdigest()}


@contextmanager
def exclusive_lock(path):
    """Advisory process lock; a killed process releases it automatically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'another writer holds {path}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def publish_bundle(destination, completion_file='manifest.json'):
    """Keep failed attempts, and expose the final name only after validation.

    Caller writes and validates in the yielded sibling directory. Directory rename
    is atomic on the same filesystem. This is process-interruption protection,
    not a guarantee of durability after power loss.
    """
    destination = Path(destination)
    with exclusive_lock(destination.parent / f'.{destination.name}.lock'):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.incomplete-',
                                          dir=destination.parent))
        try:
            yield temporary
            if not (temporary / completion_file).is_file():
                raise ValueError(f'bundle has no {completion_file}')
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            temporary.rename(destination)
        except BaseException as error:
            try:
                write_json(temporary / 'failure.json', {
                    'state': 'incomplete', 'error_type': type(error).__name__,
                    'error': str(error), 'time_utc': datetime.now(timezone.utc).isoformat(),
                })
            except OSError:
                pass  # Preserve the original error even if storage is full.
            raise
