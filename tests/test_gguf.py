import struct
import numpy as np
import pytest
from recipe_lab.gguf import inspect_gguf, decode_rows, encode_rows, compile_gguf
from recipe_lab.linear import DirectionPatch
from recipe_lab.storage import sha256_file


def string(s):
    b=s.encode();return struct.pack('<Q',len(b))+b


def packed_fixture(qtype, rows=4, cols=32):
    if qtype in (0,1,30):
        x=(np.arange(rows*cols).reshape(rows,cols)%9-4).astype(np.float32)/8
        return encode_rows(x,b'',qtype)[0]
    size=18 if qtype==2 else 34
    b=np.zeros((rows*cols//32,size),dtype=np.uint8)
    scales=np.full((len(b),1),.125,dtype='<f2')
    if qtype==2: scales[::2]*=-1  # Q4_0 can have negative scales
    b[:,:2]=scales.view(np.uint8)
    if qtype==2:
        codes=np.tile(np.arange(32)%16,(len(b),1)).astype(np.uint8)
        b[:,2:]=codes[:,:16] | (codes[:,16:]<<4)
    elif qtype==8:
        codes=np.tile(np.arange(32)-16,(len(b),1)).astype(np.int8)
        b[:,2:]=codes.view(np.uint8)
    return b.tobytes()


def make_gguf(path,qtype,split=1):
    payload=packed_fixture(qtype if qtype in (0,1,2,8,30) else 0)
    keep=np.arange(32,dtype='<f4').tobytes()
    meta=string('general.architecture')+struct.pack('<I',8)+string('synthetic_fixture_not_a_model')
    meta+=string('split.count')+struct.pack('<I',4)+struct.pack('<I',split)
    # test metadata arrays are skipped, not assumed absent
    meta+=string('test.array')+struct.pack('<I',9)+struct.pack('<IQ',8,2)+string('a')+string('b')
    header=b'GGUF'+struct.pack('<IQQ',3,2,3)+meta
    info=string('blk.0.attn_output.weight')+struct.pack('<IQQIQ',2,32,4,qtype,0)
    keep_offset=(len(payload)+31)//32*32
    info+=string('keep.weight')+struct.pack('<IQQIQ',2,32,1,0,keep_offset)
    prefix=header+info;prefix+=b'\0'*((-len(prefix))%32)
    path.write_bytes(prefix+payload+b'\0'*(keep_offset-len(payload))+keep)
    return payload


@pytest.mark.parametrize('qtype',[0,1,2,8,30])
def test_roundtrip_codec(qtype):
    raw=packed_fixture(qtype)
    x=decode_rows(raw,qtype,32)
    encoded,clip,zero=encode_rows(x,raw,qtype)
    assert encoded==raw and clip==zero==0


@pytest.mark.parametrize('qtype',[0,1,2,8,30])
def test_streaming_compiler_keeps_other_bytes(tmp_path,qtype):
    src=tmp_path/'src.gguf';dst=tmp_path/'dst.gguf';raw=make_gguf(src,qtype)
    before=src.read_bytes();sha=sha256_file(src)
    d=np.array([1.,2.,1.,-1.]);d/=np.linalg.norm(d)
    p=DirectionPatch(d[:,None],[.1])
    rep=compile_gguf(src,dst,{'blk.0.attn_output.weight':p},sha,tile_rows=1,max_clip_fraction=1)
    info=inspect_gguf(src);t=info.tensors['blk.0.attn_output.weight']
    after=dst.read_bytes()
    assert before[:t.offset]==after[:t.offset]
    assert before[t.offset+len(raw):]==after[t.offset+len(raw):]
    assert sha256_file(src)==sha and len(before)==len(after)
    actual=after[t.offset:t.offset+len(raw)]
    expected,_,_=encode_rows(p.weights(decode_rows(raw,qtype,32)),raw,qtype)
    assert actual==expected
    assert rep['qualified_for_model_runtime'] is False
    if qtype in (2,8):
        size=18 if qtype==2 else 34
        np.testing.assert_array_equal(np.frombuffer(actual,np.uint8).reshape(-1,size)[:,:2],
                                      np.frombuffer(raw,np.uint8).reshape(-1,size)[:,:2])


def test_unknown_target_rejected_before_copy(tmp_path):
    src=tmp_path/'src.gguf';dst=tmp_path/'dst.gguf';make_gguf(src,12)
    p=DirectionPatch(np.eye(4)[:,:1],[.1])
    with pytest.raises(ValueError,match='unsupported target'):
        compile_gguf(src,dst,{'blk.0.attn_output.weight':p},sha256_file(src))
    assert not dst.exists() and not list(tmp_path.glob('.recipe-*'))


def test_sharded_model_is_explicitly_not_implemented(tmp_path):
    src=tmp_path/'src.gguf';dst=tmp_path/'dst.gguf';make_gguf(src,8,split=9)
    with pytest.raises(ValueError,match='sharded'):
        compile_gguf(src,dst,{'blk.0.attn_output.weight':DirectionPatch(np.eye(4),[0]*4)},sha256_file(src))
    assert not dst.exists()


def test_identity_byte_identical(tmp_path):
    src=tmp_path/'src.gguf';dst=tmp_path/'dst.gguf';make_gguf(src,8)
    compile_gguf(src,dst,{'blk.0.attn_output.weight':DirectionPatch(np.eye(4),[0]*4)},sha256_file(src))
    assert src.read_bytes()==dst.read_bytes()


def test_clipping_aborts_atomically(tmp_path):
    src=tmp_path/'src.gguf';dst=tmp_path/'dst.gguf';make_gguf(src,8)
    with pytest.raises(ValueError,match='clipping'):
        compile_gguf(src,dst,{'blk.0.attn_output.weight':DirectionPatch(np.eye(4),[-100]*4)},sha256_file(src),max_clip_fraction=0)
    assert not dst.exists() and not list(tmp_path.glob('.recipe-*'))


def test_zero_scale_is_not_magically_fixed():
    raw=bytes(34)
    _,_,zero=encode_rows(np.ones((1,32)),raw,8)
    assert zero==32


def test_named_companion_bias_rejected_before_copy(tmp_path):
    src, dst = tmp_path/'biased.gguf', tmp_path/'out.gguf'
    payload = packed_fixture(0)
    bias_offset = (len(payload)+31)//32*32
    prefix = b'GGUF'+struct.pack('<IQQ',3,2,0)
    prefix += string('blk.0.attn_output.weight')+struct.pack('<IQQIQ',2,32,4,0,0)
    prefix += string('blk.0.attn_output.bias')+struct.pack('<IQIQ',1,4,0,bias_offset)
    prefix += b'\0' * ((-len(prefix))%32)
    src.write_bytes(prefix+payload+b'\0'*(bias_offset-len(payload))+np.ones(4,dtype='<f4').tobytes())
    patch = DirectionPatch(np.eye(4)[:,:1],[.1])
    with pytest.raises(ValueError,match='companion bias'):
        compile_gguf(src,dst,{'blk.0.attn_output.weight':patch},sha256_file(src))
    assert not dst.exists()


def test_q4_known_block_decodes_low_then_high_nibbles():
    # Native block layout: fp16 d, then 16 bytes; low half before high half.
    raw = struct.pack('<e',-0.5) + bytes([0xF0]*16)
    expected = np.array([4.0]*16 + [-3.5]*16,dtype=np.float32)[None,:]
    np.testing.assert_array_equal(decode_rows(raw,2,32),expected)
    assert encode_rows(expected,raw,2)[0] == raw


def test_q8_known_block_signed_payload():
    raw = struct.pack('<e',0.25)+np.array([-2,-1,0,1]+[2]*28,dtype=np.int8).tobytes()
    expected = np.array([-.5,-.25,0,.25]+[.5]*28,dtype=np.float32)[None,:]
    np.testing.assert_array_equal(decode_rows(raw,8,32),expected)
    assert encode_rows(expected,raw,8)[0] == raw
