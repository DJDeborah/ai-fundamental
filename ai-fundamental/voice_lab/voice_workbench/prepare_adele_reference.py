"""Fetch a bounded reference from an original publisher's public media.

References and transcripts remain in ignored assets. No browser cookies,
login bypass, copyrighted audio redistribution or open-license assertion.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
SOURCE_PAGE = "https://www.vogue.com/video/watch/73-questions-with-adele"
YOUTUBE_PAGE = "https://www.youtube.com/watch?v=544DTGHIBM0"
ASSET_ROOT = (ROOT / 'assets').resolve()
CATALOG = ASSET_ROOT / 'reference_catalog.json'
CATALOG_ID_PATTERN = r'[a-z0-9][a-z0-9_-]{0,63}'


def validate_catalog(payload):
    """Validate paths and identities without discovering additional private files."""
    if not isinstance(payload, dict) or not isinstance(payload.get('items'), list):
        raise ValueError('Reference catalog must be an object with an items array')
    ids = set()
    for item in payload['items']:
        if not isinstance(item, dict):
            raise ValueError('Each catalog item must be an object')
        reference_id = item.get('id')
        if not isinstance(reference_id, str) or not re.fullmatch(CATALOG_ID_PATTERN, reference_id):
            raise ValueError('Catalog ids must be 1-64 lowercase letters, digits, underscores or hyphens')
        if reference_id in ids:
            raise ValueError('Duplicate catalog id')
        ids.add(reference_id)
        for field in ('name', 'path', 'source_url', 'license'):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError('Catalog item is missing ' + field)
        relative = Path(item['path'])
        if relative.is_absolute():
            raise ValueError('Catalog path must be relative to assets')
        destination = (ASSET_ROOT / relative).resolve()
        try:
            destination.relative_to(ASSET_ROOT)
        except ValueError as exc:
            raise ValueError('Catalog path is outside assets') from exc
        if destination.suffix.lower() != '.wav':
            raise ValueError('Catalog references must be WAV files')
        source_url = urlparse(item['source_url'])
        if source_url.scheme not in ('http', 'https') or not source_url.netloc:
            raise ValueError('Catalog source_url must be an HTTP(S) source')
        if item.get('kind') not in ('speech', 'singing'):
            raise ValueError('Catalog kind must be speech or singing')
        if not isinstance(item.get('quality_approved', False), bool):
            raise ValueError('quality_approved must be a boolean')


def register_reference(reference_id, destination, kind, license_text, *, source_url=None, display_name=None):
    """Replace this id only, retain other entries and atomically install the JSON."""
    if not re.fullmatch(CATALOG_ID_PATTERN, reference_id):
        raise ValueError('Invalid catalog id')
    destination = destination.resolve()
    if not destination.is_file():
        raise ValueError('Cannot register a missing reference')
    relative = destination.relative_to(ASSET_ROOT).as_posix()
    payload = json.loads(CATALOG.read_text(encoding='utf-8-sig')) if CATALOG.is_file() else {'items': []}
    validate_catalog(payload)
    item = {'id': reference_id,
            'name': display_name if display_name is not None else
                    'Adele · Vogue ' + ('现场清唱候选' if kind == 'singing' else '讲话候选'),
            'path': relative, 'kind': kind,
            'source_url': source_url if source_url is not None else SOURCE_PAGE,
            'license': license_text, 'quality_approved': False}
    validate_catalog({'items': [item]})
    replaced = False
    updated = []
    for existing in payload['items']:
        if existing['id'] == reference_id:
            updated.append(item)
            replaced = True
        else:
            updated.append(existing)
    if not replaced:
        updated.append(item)
    payload['items'] = updated
    validate_catalog(payload)
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=CATALOG.parent,
                                         prefix='.reference_catalog-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CATALOG)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def selection_basis(start, duration, kind):
    if kind == 'singing' and math.isclose(start, 637.234, abs_tol=1e-6) and math.isclose(duration, 6.316, abs_tol=1e-6):
        return 'Official timed caption marks live singing at 10:37.234-10:43.550, before app-effect playback.'
    return ('User-selected crop time; this script has not independently verified who is speaking or singing '
            'throughout this interval. Validate speaker turns and background by listening.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=float, default=637.234)
    parser.add_argument('--duration', type=float, default=6.316)
    parser.add_argument('--name', default='adele_vogue_live_singing_candidate')
    parser.add_argument('--kind', choices=('speech', 'singing'), default='singing')
    parser.add_argument('--catalog-id', help='Optional local reference-catalog id; replace only that id after a valid download')
    args = parser.parse_args()
    if not math.isfinite(args.start) or not math.isfinite(args.duration) or not 0 < args.duration <= 30 or args.start < 0:
        raise ValueError('Reference must be between 0 and 30 seconds')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', args.name):
        raise ValueError('Use a simple reference name')
    if args.catalog_id is not None and not re.fullmatch(CATALOG_ID_PATTERN, args.catalog_id):
        raise ValueError('Use a unique simple lowercase catalog id')
    import yt_dlp
    with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True, 'socket_timeout': 25,
                           'retries': 0, 'skip_download': True}) as downloader:
        info = downloader.extract_info(SOURCE_PAGE, download=False)
    if info.get('id') != '616ec24838d06964439f4946':
        raise RuntimeError('Publisher video identity changed')
    formats = [f for f in info['formats'] if f['format_id'].startswith('hls-')]
    media = min(formats, key=lambda f: f.get('tbr') or 100000)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpeg not available')
    folder = ROOT / 'assets' / 'adele_private_reference'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / (args.name + '.wav')
    result = subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-ss', str(args.start), '-i', media['url'], '-t', str(args.duration),
        '-vn', '-ac', '1', '-ar', '22050', '-c:a', 'pcm_s16le', '-y', str(destination)],
        capture_output=True, timeout=100)
    if result.returncode:
        raise RuntimeError('Public publisher stream could not be cropped; no authentication workaround attempted')
    import numpy as np
    import soundfile as sf
    wave, rate = sf.read(destination)
    if not len(wave) or not np.isfinite(wave).all():
        raise RuntimeError('Reference audio is invalid')
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'speaker': 'Adele', 'publisher': 'Vogue / Conde Nast',
        'source_page': SOURCE_PAGE, 'same_publisher_youtube': YOUTUBE_PAGE,
        'source_video_id': info['id'], 'source_duration_seconds': info.get('duration'),
        'source_start_seconds': args.start, 'requested_seconds': args.duration,
        'kind': args.kind,
        'selection_basis': selection_basis(args.start, args.duration, args.kind),
        'selection_caveat': 'Timed captions identify a candidate turn, not a listening approval or proof of clean background.',
        'sample_rate': rate, 'seconds': len(wave)/rate,
        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'peak': float(np.max(np.abs(wave))), 'digital_clip_fraction': float(np.mean(np.abs(wave)>=.99)),
        'license': 'Copyrighted public interview; not an openly licensed voice library',
        'separation_applied': False, 'background_quality_approved': False,
        'publicly_reuploaded': False, 'purpose': 'Local reference-conditioned experiment requested by user'}
    destination.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    if args.catalog_id is not None:
        register_reference(args.catalog_id, destination, args.kind, receipt['license'])
    print(json.dumps({'reference': str(destination), 'seconds': receipt['seconds'],
                      'sha256': receipt['sha256'], 'quality_approved': False,
                      'catalog_id': args.catalog_id, 'catalog_registered': args.catalog_id is not None}))


if __name__ == '__main__':
    main()
