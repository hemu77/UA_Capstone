"""Bounded, hash-pinned public graph parsing. Never extracts archive paths."""
import gzip
import hashlib
import io
from pathlib import Path, PurePosixPath
import tarfile

import networkx as nx

PROTOCOL_SHA256 = 'eeab443106fa389e66bdf88d0b71d884d4da1815aefe9cf57269b13209f51d2a'
SOURCES = {
    'snap_facebook': dict(filename='facebook.tar.gz', sha256='78594c43169b698e99a77741bc298581af87146e4f3d3e8ad48a48361b5d92b6',
                          url='https://snap.stanford.edu/data/facebook.tar.gz'),
    'sociopatterns_friendship': dict(filename='HighSchool2013_friendship_network.csv.gz',
        sha256='22b2c9d09bfca93a18962698918ad58dce683952dc4fce4c69a9fc03d3c11713',
        url='https://sociopatterns.org/assets/data/HighSchool2013_friendship_network.csv.gz'),
}
MAX_COMPRESSED = 20_000_000
MAX_EXPANDED = 60_000_000


def unpack_gzip(data):
    if len(data) > MAX_COMPRESSED:
        raise ValueError('Compressed public input exceeds bound')
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        result = stream.read(MAX_EXPANDED+1)
    if len(result) > MAX_EXPANDED:
        raise ValueError('Expanded public input exceeds bound')
    return result


def read_pinned(cache, source):
    info = SOURCES[source]
    path = cache/info['filename']
    if path.stat().st_size > MAX_COMPRESSED:
        raise ValueError('Public input exceeds bound')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != info['sha256']:
        raise ValueError('Public source SHA-256 differs: '+source)
    return unpack_gzip(raw)


def facebook_graphs(tar_bytes):
    result = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes),mode='r:') as archive:
        members = archive.getmembers()
        if len(members) > 100 or sum(x.size for x in members) > MAX_EXPANDED:
            raise ValueError('Facebook archive inventory exceeds bound')
        selected = {}
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise ValueError('Unsafe public archive member')
            if member.isfile() and path.suffix in {'.edges','.feat'}:
                if len(path.parts) != 2 or path.parts[0] != 'facebook' or member.size > 8_000_000:
                    raise ValueError('Unexpected Facebook member')
                if path.name in selected:
                    raise ValueError('Duplicate public archive member')
                selected[path.name] = archive.extractfile(member).read()
        egos = sorted((int(Path(name).stem) for name in selected if name.endswith('.edges')))
        if len(egos) != 10 or len(selected) != 20:
            raise ValueError('Require exactly ten Facebook edge/roster pairs')
        for index, ego in enumerate(egos,1):
            roster = [int(line.split()[0]) for line in selected[f'{ego}.feat'].decode().splitlines() if line.strip()]
            if len(roster) != len(set(roster)) or ego in roster:
                raise ValueError('Duplicate roster node or included ego')
            graph = nx.Graph()
            graph.add_nodes_from(sorted(roster))
            raw_edges = 0
            for line in selected[f'{ego}.edges'].decode().splitlines():
                if not line.strip(): continue
                pair = line.split()
                if len(pair) != 2: raise ValueError('Expected two Facebook endpoints')
                a,b = map(int,pair)
                if a not in graph or b not in graph or a == b:
                    raise ValueError('Unknown Facebook endpoint or self-loop')
                graph.add_edge(a,b)
                raw_edges += 1
            graph.graph.update(source='snap_facebook',cohort=f'FB{index:02}',member=f'facebook/{ego}.edges',
                roster='first-column feature IDs only; ego excluded; no profile attributes used',
                raw_directed_rows=raw_edges,dropped_external_nominations=0)
            result.append(graph)
    return result


def school_graph(data):
    pairs=[]
    for line in data.decode().splitlines():
        if not line.strip(): continue
        tokens=line.split()
        if len(tokens) != 2: raise ValueError('Expected two survey endpoints')
        a,b=map(int,tokens)
        if a==b: raise ValueError('Unexpected survey self-nomination')
        pairs.append((a,b))
    roster={a for a,b in pairs}
    graph=nx.Graph()
    graph.add_nodes_from(sorted(roster))
    graph.add_edges_from((a,b) for a,b in pairs if b in roster)
    graph.graph.update(source='sociopatterns_friendship',cohort='School2013',member=SOURCES['sociopatterns_friendship']['filename'],
        roster='positive-reporting nominators only; restrict nominations to this observed roster; OR projection',
        raw_directed_rows=len(pairs),dropped_external_nominations=sum(b not in roster for a,b in pairs))
    return graph


def load_public(cache):
    graphs=facebook_graphs(read_pinned(cache,'snap_facebook'))
    graphs.append(school_graph(read_pinned(cache,'sociopatterns_friendship')))
    if any(len(g)<50 for g in graphs):
        raise ValueError('Planned cohort smaller than 50: do not silently change sampling')
    return graphs
