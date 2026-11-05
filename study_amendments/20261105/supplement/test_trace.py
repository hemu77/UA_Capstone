import contextlib
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parent
def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

t, s = load('traced_launch'), load('runner')
class AuthenticationError(Exception):
    status_code = 401
    code = 'invalid_api_key'

output = io.StringIO()
with patch.object(s, 'launch', side_effect=AuthenticationError('PRIVATE_SENTINEL')), patch.object(s.sys, 'argv', ['runner.py', 'launch']), contextlib.redirect_stdout(output):
    code, diagnostic = t.run(s)
assert code == 1 and diagnostic['http_status'] == 401 and diagnostic['error_code'] == 'invalid_api_key'
assert 'PRIVATE_SENTINEL' not in output.getvalue() + str(diagnostic)
with patch.object(AuthenticationError, 'code', {'PRIVATE_SENTINEL': 'untrusted'}), patch.object(s, 'launch', side_effect=AuthenticationError('PRIVATE_SENTINEL')), patch.object(s.sys, 'argv', ['runner.py', 'launch']), contextlib.redirect_stdout(output):
    code, diagnostic = t.run(s)
assert code == 1 and diagnostic['error_code'] is None and 'PRIVATE_SENTINEL' not in str(diagnostic)
with patch.object(s, 'launch', return_value=None), patch.object(s.sys, 'argv', ['runner.py', 'launch']), contextlib.redirect_stdout(output):
    code, diagnostic = t.run(s)
assert code == 0
print('TRACE_REDACTION_TESTS_PASSED; no provider calls')
