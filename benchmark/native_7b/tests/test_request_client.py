"""Exercise native response parsing without running inference or opening sockets."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2]/'send_request/request.py'
spec = importlib.util.spec_from_file_location('native_request_client', SOURCE)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class Response:
    def __init__(self, status, payload):
        self.status, self.payload = status, payload

    async def json(self):
        return self.payload


class Session:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, endpoint, json):
        self.endpoint, self.payload = endpoint, json
        return self.response


class RequestClientTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, status, payload):
        session = Session(Response(status, payload))
        with patch.object(client.aiohttp, 'ClientSession', return_value=session):
            return await client.request_head_node({'params': {}}, 'http://example.invalid', 'test')

    async def test_json_response_with_unicode(self):
        result = await self.call(200, {'data': json.dumps(['true, null and Unicode: café', 1.25])})
        self.assertEqual(result[:2], ('true, null and Unicode: café', 1.25))

    async def test_non_success_status_preserves_error(self):
        with self.assertRaisesRegex(RuntimeError, 'HTTP 502.*stream limit'):
            await self.call(502, {'error': 'stream limit'})

    async def test_error_in_success_status_is_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'HTTP 200.*worker failed'):
            await self.call(200, {'error': 'worker failed'})

    async def test_python_expression_is_not_evaluated(self):
        with self.assertRaises(json.JSONDecodeError):
            await self.call(200, {'data': "('hello', 1.0)"})


if __name__ == '__main__':
    unittest.main()
