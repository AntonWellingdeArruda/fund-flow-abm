import pytest

from fund_flow.config import has_secret
from fund_flow.data.anbima import AnbimaClient, AnbimaError
from fund_flow.data.sources import AnbimaFlowSource


class TestFlowSourceGuards:
    def test_load_without_path_raises_helpful(self):
        # OAuth is wired, but the data endpoint is still unknown.
        with pytest.raises(NotImplementedError, match="data endpoint"):
            AnbimaFlowSource().load()


class TestTokenCache:
    def test_cached_token_not_refetched(self, monkeypatch):
        client = AnbimaClient("id", "secret")
        calls = {"n": 0}

        def fake_fetch():
            calls["n"] += 1
            return "tok123", 3600

        monkeypatch.setattr(client, "_fetch_token", fake_fetch)
        assert client.access_token() == "tok123"
        assert client.access_token() == "tok123"   # served from cache
        assert calls["n"] == 1

    def test_refetch_after_expiry(self, monkeypatch):
        client = AnbimaClient("id", "secret")
        seq = iter([("a", 3600), ("b", 3600)])
        monkeypatch.setattr(client, "_fetch_token", lambda: next(seq))
        assert client.access_token() == "a"
        client._expires_at = 0          # force expiry
        assert client.access_token() == "b"


# --- Network + secret gated live OAuth ---

@pytest.fixture(scope="module")
def anbima_creds():
    if not (has_secret("ANBIMA_CLIENT_ID") and has_secret("ANBIMA_CLIENT_SECRET")):
        pytest.skip("ANBIMA credentials not set")
    from fund_flow.config import get_secret
    return get_secret("ANBIMA_CLIENT_ID"), get_secret("ANBIMA_CLIENT_SECRET")


@pytest.mark.live
class TestLiveOAuth:
    def test_real_token_exchange(self, anbima_creds):
        cid, secret = anbima_creds
        client = AnbimaClient(cid, secret)
        try:
            token = client.access_token()
        except AnbimaError:
            pytest.skip("ANBIMA OAuth unreachable (offline)")
        assert isinstance(token, str) and len(token) > 0
