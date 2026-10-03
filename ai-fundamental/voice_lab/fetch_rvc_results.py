"""Retrieve this experiment through authenticated SSH; never store a password."""
import argparse
import datetime as dt
import getpass
import hashlib
import json
import pathlib
import shlex
import time
import zipfile

import paramiko


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--root', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--retrieve-now', action='store_true', help='Retrieve existing results only; no training or waiting')
    p.add_argument('--resume', action='store_true', help='Append the missing tail of an interrupted archive')
    a = p.parse_args()
    deadline = dt.datetime.fromisoformat(a.deadline)
    destination = pathlib.Path(__file__).resolve().parent / 'runs' / a.run
    destination.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass('SSH password (not stored): ')
    client.connect(a.host, port=a.port, username='root', password=password,
                   look_for_keys=False, allow_agent=False, timeout=20)
    del password
    client.get_transport().set_keepalive(30)
    sftp = paramiko.SFTPClient.from_transport(client.get_transport(),
             window_size=64 * 1024 * 1024, max_packet_size=32768)
    root = a.root.rstrip('/')
    output = f'{root}/runs/{a.run}'
    log = f'{root}/{a.log}'
    print('Authenticated; waiting for finalized results.', flush=True)
    last = {}
    try:
        while not a.retrieve_now and dt.datetime.now(dt.timezone.utc) < deadline:
            with sftp.open(log) as f:
                f.seek(max(0, f.stat().st_size - 16000))
                tail = f.read().decode('utf-8', errors='replace')
            for arm in ('scratch', 'finetune'):
                try:
                    with sftp.open(f'{output}/{arm}/metrics.jsonl') as f:
                        f.seek(max(0, f.stat().st_size - 4000))
                        rows = f.read().decode().strip().splitlines()
                    step = json.loads(rows[-1])['step']
                    if step >= last.get(arm, 0) + 200 or step == 1000 and last.get(arm) != 1000:
                        print(json.dumps({'arm': arm, 'step': step}), flush=True)
                        last[arm] = step
                except (FileNotFoundError, IndexError, ValueError):
                    pass
            if 'RESULTS_READY ' in tail:
                print('RESULTS_READY; retrieving finalized archive now.', flush=True)
                break
            if 'Pipeline failed; logs available' in tail:
                sftp.get(log, str(destination / a.log))
                for name in ('pipeline_failure.json', 'comparison.json', 'REPORT.md'):
                    try:
                        sftp.get(f'{output}/{name}', str(destination / name))
                    except FileNotFoundError:
                        pass
                raise RuntimeError('Pipeline failed; failure records downloaded. Diagnose before retrying.')
            time.sleep(10)
        else:
            if not a.retrieve_now:
                raise TimeoutError('Authorized deadline reached without finalized results')
        if a.retrieve_now:
            print('Retrieving existing results only; no training command.', flush=True)
        # Download the small audit files first while the cloud is still accessible.
        for source, name in [(log, a.log), (f'{root}/runs/{a.run}_billing.json', 'billing_at_download.json'),
                             (f'{root}/NEW_INSTANCE_RETRY1.json', 'NEW_INSTANCE_RETRY1.json')]:
            try:
                sftp.get(source, str(destination / name))
            except FileNotFoundError:
                pass
        archive = f'{output}/results.zip'
        _, stdout, stderr = client.exec_command('sha256sum -- ' + shlex.quote(archive), timeout=15)
        expected = stdout.read().decode().split()[0]
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError('Remote checksum failed: ' + stderr.read().decode())
        target = destination / 'results.zip'
        if a.resume and not target.exists() and (destination / 'results.zip.partial').exists():
            target = destination / 'results.zip.partial'
        remote_size = sftp.stat(archive).st_size
        (destination / 'DOWNLOAD_EXPECTED.json').write_text(json.dumps({'remote_sha256': expected,
            'remote_bytes': remote_size, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2), encoding='utf-8')
        started = time.monotonic()
        offset = target.stat().st_size if a.resume and target.exists() else 0
        if offset > remote_size:
            raise RuntimeError('Local archive is larger than the remote archive')
        if offset:
            print(json.dumps({'resume_offset': offset, 'remaining_bytes': remote_size - offset}), flush=True)
            with sftp.open(archive, 'rb') as source, target.open('ab') as sink:
                source.seek(offset)
                source.prefetch(remote_size)
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
        else:
            sftp.get(archive, str(target))
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Downloaded archive checksum mismatch')
        if target.name.endswith('.partial'):
            target = target.replace(destination / 'results.zip')
        with zipfile.ZipFile(destination / 'results.zip') as z:
            for item in z.infolist():
                target = (destination / item.filename).resolve()
                if not target.is_relative_to(destination.resolve()) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Unsafe archive member')
            z.extractall(destination)
        receipt = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'remote': archive,
                   'local': str(destination), 'sha256': actual,
                   'bytes': (destination / 'results.zip').stat().st_size,
                   'download_seconds': time.monotonic() - started,
                   'credentials_saved': False}
        (destination / 'DOWNLOAD_RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt), flush=True)
    finally:
        sftp.close()
        client.close()


if __name__ == '__main__':
    main()
