"""LCM type definitions
This file automatically generated by pose_and_cls_t.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class pose_and_cls_t(object):
    __slots__ = ["timestamp", "position", "orientation", "name"]

    __typenames__ = ["int64_t", "double", "double", "string"]

    __dimensions__ = [None, [3], [4], None]

    def __init__(self):
        self.timestamp = 0
        self.position = [ 0.0 for dim0 in range(3) ]
        self.orientation = [ 0.0 for dim0 in range(4) ]
        self.name = ""

    def encode(self):
        buf = BytesIO()
        buf.write(pose_and_cls_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">q", self.timestamp))
        buf.write(struct.pack('>3d', *self.position[:3]))
        buf.write(struct.pack('>4d', *self.orientation[:4]))
        __name_encoded = self.name.encode('utf-8')
        buf.write(struct.pack('>I', len(__name_encoded)+1))
        buf.write(__name_encoded)
        buf.write(b"\0")

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != pose_and_cls_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return pose_and_cls_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = pose_and_cls_t()
        self.timestamp = struct.unpack(">q", buf.read(8))[0]
        self.position = struct.unpack('>3d', buf.read(24))
        self.orientation = struct.unpack('>4d', buf.read(32))
        __name_len = struct.unpack('>I', buf.read(4))[0]
        self.name = buf.read(__name_len)[:-1].decode('utf-8', 'replace')
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if pose_and_cls_t in parents: return 0
        tmphash = (0xebfd7106d35559d8) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if pose_and_cls_t._packed_fingerprint is None:
            pose_and_cls_t._packed_fingerprint = struct.pack(">Q", pose_and_cls_t._get_hash_recursive([]))
        return pose_and_cls_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

