"""Run real recorded-clip reference conversions through the local HTTP API."""
import argparse
import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path


def request_json(url):
    with urllib.request.urlopen(url, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--self-reference', type=Path)
    parser.add_argument('--reference-id', required=True)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--mode', choices=('speech', 'singing'), default='singing')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.alpha <= 1:
        raise ValueError('This model benchmark requires 0 < alpha <= 1')
    base = 'http://127.0.0.1:8872'
    boundary = 'voice-' + uuid.uuid4().hex
    parts = []
    def field(name, value):
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode())
    def upload(name, path):
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="recording.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode())
        parts.extend([path.read_bytes(), b'\r\n'])
    upload('source_file', args.source)
    if args.self_reference:
        upload('self_file', args.self_reference)
    for name, value in {'reference_id': args.reference_id, 'alpha': args.alpha,
                        'mode': args.mode, 'method': 'lerp'}.items():
        field(name, value)
    parts.append(f'--{boundary}--\r\n'.encode())
    started = time.perf_counter()
    request = urllib.request.Request(base + '/api/convert', data=b''.join(parts),
        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    if result.get('state') != 'complete' or result.get('result', {}).get('ai_inference') is not True:
        raise RuntimeError('No completed model inference returned')
    receipt = request_json(base + result['receipt_url'])
    with urllib.request.urlopen(base + result['audio_url'], timeout=30) as response:
        audio = response.read()
    digest = hashlib.sha256(audio).hexdigest()
    if digest != receipt['output']['sha256']:
        raise RuntimeError('Downloaded output SHA256 mismatch')
    if receipt.get('reference_id') != args.reference_id:
        raise RuntimeError('The model used a different reference')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(audio)
    args.out.with_suffix('.receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.out), 'reference_id': args.reference_id,
                      'alpha': args.alpha, 'seconds': receipt['output']['duration_s'],
                      'sha256': digest, 'http_wall_s': time.perf_counter()-started,
                      'quality_approved': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
