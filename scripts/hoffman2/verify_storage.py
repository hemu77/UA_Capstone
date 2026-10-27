"""Two-node storage check in a disposable directory, never in a study ledger."""
import fcntl
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import time


def wait_for(path):
    deadline = time.monotonic() + 90
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(str(path))
        time.sleep(0.1)


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    root = Path(sys.argv[1]).resolve()
    rank = int(os.environ['SLURM_PROCID'])
    assert int(os.environ['SLURM_NTASKS']) == 2 and rank in (0, 1)
    assert root.is_dir()
    host = socket.gethostname()
    lock = root/'probe.lock'
    database = root/'probe.sqlite'
    if rank == 0:
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with sqlite3.connect(database, timeout=0) as db:
                db.execute('CREATE TABLE probe (value TEXT NOT NULL)')
                db.commit()
                db.execute('BEGIN IMMEDIATE')
                db.execute('INSERT INTO probe VALUES (?)', ('committed',))
                publish(root/'locked.json', {'host': host})
                wait_for(root/'contended.json')
                other = json.loads((root/'contended.json').read_text())
                assert other['host'] != host, 'Probe needs two distinct compute nodes'
                db.commit()
            publish(root/'atomic.json', {'value': 'old'})
            publish(root/'atomic.json', {'value': 'new'})
        publish(root/'released.json', {'released': True})
        wait_for(root/'verified.json')
        print((root/'verified.json').read_text())
    else:
        wait_for(root/'locked.json')
        with lock.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise AssertionError('Cross-node flock allowed a second owner')
        with sqlite3.connect(database, timeout=0) as db:
            try:
                db.execute('BEGIN IMMEDIATE')
            except sqlite3.OperationalError as error:
                assert 'locked' in str(error).lower(), str(error)
            else:
                raise AssertionError('SQLite allowed concurrent write transactions')
        publish(root/'contended.json', {'host': host})
        wait_for(root/'released.json')
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with sqlite3.connect(database, timeout=0) as db:
                db.execute('BEGIN IMMEDIATE')
                assert db.execute('SELECT value FROM probe').fetchall() == [('committed',)]
                db.execute('INSERT INTO probe VALUES (?)', ('resumed',))
            assert json.loads((root/'atomic.json').read_text()) == {'value': 'new'}
        first = json.loads((root/'locked.json').read_text())['host']
        publish(root/'verified.json', dict(status='CROSS_NODE_STORAGE_PASSED', job_id=os.environ['SLURM_JOB_ID'],
            hosts=[first, host], checks=['exclusive_flock', 'sqlite_contention',
                'sqlite_after_release', 'atomic_replace_visibility'], paid_calls=0))


if __name__ == '__main__':
    main()
