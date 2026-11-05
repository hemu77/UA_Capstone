"""Fetch two public research inputs only; pinned SHA-256, no credentials or models."""
import argparse
import hashlib
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_heldout.data import SOURCES, MAX_COMPRESSED


class SourceRedirects(HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        if urlparse(newurl).scheme!='https' or urlparse(newurl).hostname!=urlparse(request.full_url).hostname:
            raise ValueError('Cross-host or non-HTTPS source redirect refused')
        return super().redirect_request(request,fp,code,msg,headers,newurl)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    cache=args.cache.resolve()
    cache.mkdir(parents=True,exist_ok=True)
    for name,info in SOURCES.items():
        path=cache/info['filename']
        if path.exists():
            if path.stat().st_size>MAX_COMPRESSED or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:
                raise ValueError('Refuse different existing input: '+name)
            print(name+': existing SHA-256 verified',flush=True)
            continue
        request=Request(info['url'],headers={'User-Agent':'UA-Capstone-public-evaluation/1.0'})
        with build_opener(SourceRedirects()).open(request,timeout=90) as response:
            if urlparse(response.url).scheme!='https' or urlparse(response.url).hostname!=urlparse(info['url']).hostname:
                raise ValueError('Unexpected final source URL')
            data=response.read(MAX_COMPRESSED+1)
        if len(data)>MAX_COMPRESSED or hashlib.sha256(data).hexdigest()!=info['sha256']:
            raise ValueError('Downloaded source size/hash differs: '+name)
        with path.open('xb') as stream: stream.write(data)
        print(name+': downloaded and SHA-256 verified',flush=True)


if __name__=='__main__': main()
