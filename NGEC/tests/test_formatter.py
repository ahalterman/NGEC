import pytest
from ..formatter import resolve_date

def test_resolution():
    event = {"pub_date": "June 20, 2012",
            "attributes": {"DATE": [{"text": "last Sunday"}]}}
    resolve_date(event)