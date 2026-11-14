"""Fetch three bounded public files from the MIT-licensed Figshare v1 release."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data_external/copenhagen_20261114'
API = 'https://api.figshare.com/v2/articles/7267433/versions/1'
FILES = {
    'fb_friends.csv': (13389320, 49636, 'b1f96bae3e79949ed4278e90d0753843'),
    'genders.csv': (13389440, 4632, 'eb033aae23fff859684a82d00ffd8099'),
    'fb_friends.README': (13389839, 193, '86dcef05caae3c0eca1dd99f75ee6960'),
}
HOSTS = {'api.figshare.com', 'ndownloader.figshare.com',
         '44c1cec316ec56bca76dfd65d2882b20.r2.cloudflarestorage.com'}


def allowed(url):
    p = urlparse(url)
    if p.scheme != 'https' or p.hostname not in HOSTS or p.username or p.password or p.port not in (None, 443):
        raise ValueError('Unapproved public-download destination')


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        allowed(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def fetch(url, bound):
    allowed(url)
    request = Request(url, headers={'User-Agent': 'UA-Capstone-offline-evaluation/1.0'})
    with build_opener(Redirects()).open(request, timeout=90) as response:
        allowed(response.url)
        data = response.read(bound + 1)
    if len(data) > bound:
        raise ValueError('Public input exceeds declared byte bound')
    return data


def main():
    meta = json.loads(fetch(API, 100_000))
    if (meta['doi'] != '10.6084/m9.figshare.7267433.v1' or meta['license']['name'] != 'MIT'
            or not meta['is_public'] or meta['is_embargoed'] or meta['download_disabled']):
        raise ValueError('Release permission/version changed; stop for review')
    indexed = {f['name']: f for f in meta['files']}
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = dict(doi=meta['doi'], metadata_url=API, license=meta['license'],
                    authors=[a['full_name'] for a in meta['authors']], files={})
    for name, (identifier, size, md5) in FILES.items():
        f = indexed[name]
        url = f'https://ndownloader.figshare.com/files/{identifier}'
        if f['download_url'] != url or f['size'] != size or f['computed_md5'] != md5:
            raise ValueError('Release inventory changed: ' + name)
        path = CACHE / name
        if path.exists() and path.stat().st_size != size:
            raise ValueError('Existing public file differs: ' + name)
        data = path.read_bytes() if path.exists() else fetch(url, size)
        if len(data) != size or hashlib.md5(data).hexdigest() != md5:
            raise ValueError('Published size/MD5 mismatch: ' + name)
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(data)
        manifest['files'][name] = dict(url=url, bytes=size, published_md5=md5,
                                      sha256=hashlib.sha256(data).hexdigest())
        print(name + ': verified', flush=True)
    target = CACHE / 'source_manifest.json'
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    if target.exists():
        if target.read_bytes() != encoded:
            raise ValueError('Existing acquisition manifest differs')
    else:
        with target.open('xb') as stream:
            stream.write(encoded)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
