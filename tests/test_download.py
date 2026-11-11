import io
import zipfile
import pytest
from scripts.download_modelnet10 import CLASSES, download, extract, sha256, validate_dataset


def make_archive(path, extra=None):
    with zipfile.ZipFile(path, 'w') as z:
        for name in CLASSES:
            for split in ('train', 'test'):
                z.writestr(f'ModelNet10/{name}/{split}/sample.off', 'OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
        if extra:
            z.writestr(extra, 'bad')


def test_extract_and_idempotency(tmp_path):
    archive=tmp_path/'data.zip'; make_archive(archive)
    output=tmp_path/'ModelNet10'
    counts=extract(archive,output,expected_counts={'train':10,'test':10})
    assert counts=={'train':10,'test':10}
    assert extract(archive,output,expected_counts=counts)==counts
    with pytest.raises(ValueError,match='Unexpected'):
        validate_dataset(output)


@pytest.mark.parametrize('extra',['../escape', '/absolute', 'ModelNet10/../../escape'])
def test_reject_traversal(tmp_path,extra):
    archive=tmp_path/'bad.zip'; make_archive(archive,extra)
    with pytest.raises(ValueError,match='Unsafe'):
        extract(archive,tmp_path/'output',expected_counts=None)
    assert not (tmp_path/'output').exists()


def test_streamed_download_and_checksum(tmp_path,monkeypatch):
    source=tmp_path/'source.zip'; make_archive(source)
    payload=source.read_bytes()
    class Response(io.BytesIO):
        headers={'Content-Length':str(len(payload))}
        def geturl(self): return 'https://example.com/ModelNet10.zip'
    class Opener:
        def open(self,request,timeout): return Response(payload)
    monkeypatch.setattr('urllib.request.build_opener',lambda *args:Opener())
    dest=tmp_path/'cached.zip'
    assert download('https://example.com/ModelNet10.zip',dest,sha256(source))==dest
    assert dest.read_bytes()==payload
    with pytest.raises(ValueError,match='checksum mismatch'):
        download('https://example.com/ModelNet10.zip',dest,'0'*64)
    with pytest.raises(ValueError,match='SHA256'):
        download('https://example.com/ModelNet10.zip',tmp_path/'wrong.zip','0'*64)
    assert not (tmp_path/'wrong.zip').exists()
