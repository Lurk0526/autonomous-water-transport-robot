"""A fresh stopped status must retain its startup fault instead of timing out."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from tools import follow_service


class ServiceTests(unittest.TestCase):
    def test_stopped_fault_is_reported_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'work/bowl-follow/status.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(state='stopped',motion_enabled=False,
                updated_unix=time.time()+1,actuator_state='fault:关节 001 无位置反馈')))
            child=Mock();child.wait.return_value=0
            output=io.StringIO()
            with patch.object(follow_service,'ROOT',root), \
                 patch('sys.argv',['follow_service.py','start']), \
                 patch.object(follow_service.signal,'signal'), \
                 patch.object(follow_service.subprocess,'Popen',return_value=child), \
                 patch.object(follow_service,'stop_service') as stop, \
                 patch.object(follow_service.time,'sleep',side_effect=AssertionError('fault was ignored')), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(follow_service.main(),1)
            self.assertIn('关节 001 无位置反馈',output.getvalue())
            stop.assert_called_once_with()
