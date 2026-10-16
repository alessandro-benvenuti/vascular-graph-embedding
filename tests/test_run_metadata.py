from pathlib import Path

from vascular_graph_embedding.experiments.common import git_sha


def test_git_sha_without_git_executable(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repo"
    reference = repository / ".git" / "refs" / "heads" / "main"
    reference.parent.mkdir(parents=True)
    (repository / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    expected = "a" * 40
    reference.write_text(expected + "\n")

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("subprocess.check_output", unavailable)
    assert git_sha(repository) == expected
