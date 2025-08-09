
_COUNTERS = {'markitdown':0,'pp':0,'llm_chat':0,'llm_embed':0}
def reset_mock_counters():
    for k in list(_COUNTERS.keys()):
        _COUNTERS[k]=0
def get_mock_counters():
    return dict(_COUNTERS)

# Re-export runtime client functions from the sibling module 'clients.py' if available
try:
    from importlib import import_module as _import_module
    _mod = _import_module('clients')  # may resolve to this package; handle carefully
    # If this is the package (no llm_embed), try sibling module by full path hack
    if not hasattr(_mod, 'llm_embed'):
        import types, sys, os
        _file = os.path.join(os.path.dirname(__file__), '..', 'clients.py')
        _file = os.path.normpath(_file)
        ns = {}
        with open(_file, 'r', encoding='utf-8') as f:
            code = f.read()
        module = types.ModuleType('clients_module')
        exec(compile(code, _file, 'exec'), module.__dict__)
        sys.modules['clients_module'] = module
        _mod = module
    for name in ('markitdown_convert','pp_analyze_image','llm_chat','llm_embed','reset_mock_counters','get_mock_counters'):
        if hasattr(_mod, name):
            globals()[name] = getattr(_mod, name)
except Exception:
    pass

import numpy as _np
import hashlib as _hashlib

async def llm_embed(texts):
    # Deterministic 16-dim mock embedding with L2 normalization
    V = []
    for t in texts:
        h = _hashlib.sha256(t.encode("utf-8")).digest()
        vec = [((h[i]%127)/127.0) for i in range(16)]
        v = _np.array(vec, dtype="float32")
        v = v / (float(_np.linalg.norm(v)) + 1e-6)
        V.append(v.tolist())
    return V

def _llm_embed_sync(texts):
    # Synchronous wrapper for tests
    import numpy as _np, hashlib as _hashlib
    V = []
    for t in texts:
        h = _hashlib.sha256(t.encode("utf-8")).digest()
        vec = [((h[i]%127)/127.0) for i in range(16)]
        v = _np.array(vec, dtype="float32")
        v = v / (float(_np.linalg.norm(v)) + 1e-6)
        V.append(v.tolist())
    return V

# attach sync impl as __wrapped__ so retriever skips anyio.run
try:
    llm_embed.__wrapped__ = _llm_embed_sync
except Exception:
    pass
