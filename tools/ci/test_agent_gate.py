"""Unit tests for tools/ci/agent_gate.py and the review calls of gh.GhCli, against fakes.FakeGitHub (no network).

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
They pin the `agent-approval` rules: a pull request by claude[bot] or labelled `agent` fails until a trusted person's
approval of the current head commit stands; every other pull request passes at once.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from collections.abc import Sequence

import agent_gate
import gh
from fakes import FakeGitHub
from gh import Review

HEAD = "a" * 40
OLD = "b" * 40


def agent_pr(gh_: FakeGitHub, *, author: str = "claude[bot]", labels: tuple[str, ...] = ("agent",)) -> int:
    """An open pull request at HEAD, by `author`, with `labels`."""
    return gh_.add_pull_request(HEAD, author=author, labels=list(labels))


def approval(user: str = "dpologdvinity", *, sha: str = HEAD, assoc: str = "OWNER", state: str = "APPROVED") -> Review:
    """A review by `user` (association `assoc`) of commit `sha`."""
    return Review(user=user, state=state, commit_id=sha, association=assoc)


class VerdictTests(unittest.TestCase):
    def test_ordinary_pull_request_passes_without_reviews(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake, author="dpologdvinity", labels=("ci",))
        verdict = agent_gate.evaluate(fake, number)
        self.assertTrue(verdict.ok)
        self.assertIn("not an agent", verdict.reason)

    def test_agent_label_without_review_fails(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake, author="dpologdvinity")
        self.assertFalse(agent_gate.evaluate(fake, number).ok)

    def test_claude_bot_author_without_label_fails(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake, labels=())
        self.assertFalse(agent_gate.evaluate(fake, number).ok)

    def test_owner_approval_of_the_head_passes(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake)
        fake.reviews[number] = [approval()]
        self.assertTrue(agent_gate.evaluate(fake, number).ok)

    def test_member_and_collaborator_approvals_count(self) -> None:
        for assoc in ("MEMBER", "COLLABORATOR"):
            with self.subTest(assoc=assoc):
                fake = FakeGitHub()
                number = agent_pr(fake)
                fake.reviews[number] = [approval("someone", assoc=assoc)]
                self.assertTrue(agent_gate.evaluate(fake, number).ok)

    def test_approval_from_an_untrusted_association_fails(self) -> None:
        for assoc in ("NONE", "CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "MANNEQUIN"):
            with self.subTest(assoc=assoc):
                fake = FakeGitHub()
                number = agent_pr(fake)
                fake.reviews[number] = [approval("stranger", assoc=assoc)]
                self.assertFalse(agent_gate.evaluate(fake, number).ok)

    def test_approval_of_an_older_commit_fails(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake)
        fake.reviews[number] = [approval(sha=OLD)]
        verdict = agent_gate.evaluate(fake, number)
        self.assertFalse(verdict.ok)
        self.assertIn("current head", verdict.reason)

    def test_later_changes_requested_by_the_same_reviewer_withdraws_the_approval(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake)
        fake.reviews[number] = [approval(), approval(state="CHANGES_REQUESTED")]
        self.assertFalse(agent_gate.evaluate(fake, number).ok)

    def test_dismissed_approval_does_not_count(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake)
        fake.reviews[number] = [approval(state="DISMISSED")]
        self.assertFalse(agent_gate.evaluate(fake, number).ok)

    def test_a_plain_comment_after_an_approval_keeps_it(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake)
        fake.reviews[number] = [approval(), approval(state="COMMENTED")]
        self.assertTrue(agent_gate.evaluate(fake, number).ok)

    def test_another_reviewers_approval_stands_when_one_requests_changes(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake)
        fake.reviews[number] = [approval("a"), approval("b", state="CHANGES_REQUESTED")]
        self.assertTrue(agent_gate.evaluate(fake, number).ok)

    def test_the_author_cannot_approve_their_own_pull_request(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake, author="dpologdvinity")
        fake.reviews[number] = [approval("dpologdvinity")]
        self.assertFalse(agent_gate.evaluate(fake, number).ok)

    def test_a_bot_reviewer_does_not_count(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake)
        fake.reviews[number] = [approval("claude[bot]", assoc="COLLABORATOR")]
        self.assertFalse(agent_gate.evaluate(fake, number).ok)

    def test_login_comparison_ignores_case(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake, author="Claude[bot]", labels=())
        self.assertFalse(agent_gate.evaluate(fake, number).ok)


class MainTests(unittest.TestCase):
    def run_main(self, fake: FakeGitHub, number: int) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = agent_gate.main(["--pr", str(number)], gh=fake, repo="o/r")
        return code, out.getvalue()

    def test_exit_status_follows_the_verdict(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake)
        code, text = self.run_main(fake, number)
        self.assertEqual(code, 1)
        self.assertIn("agent-approval: fail", text)
        fake.reviews[number] = [approval()]
        code, text = self.run_main(fake, number)
        self.assertEqual(code, 0)
        self.assertIn("agent-approval: pass", text)

    def test_api_error_fails_the_check_instead_of_passing_it(self) -> None:
        fake = FakeGitHub()
        number = agent_pr(fake)
        fake.fail_reviews = True
        with self.assertRaises(gh.GhError):
            agent_gate.main(["--pr", str(number)], gh=fake, repo="o/r")


class GhCliReviewTests(unittest.TestCase):
    def test_pull_request_reads_author_labels_and_head(self) -> None:
        body = {"user": {"login": "claude[bot]"}, "labels": [{"name": "agent"}], "head": {"sha": HEAD}}
        calls: list[list[str]] = []

        def runner(argv: Sequence[str], stdin: str | None) -> str:
            calls.append(list(argv))
            return json.dumps(body)

        pr = gh.GhCli("o/r", runner).pull_request(7)
        self.assertEqual((pr.author, pr.labels, pr.head_sha), ("claude[bot]", ["agent"], HEAD))
        self.assertIn("repos/o/r/pulls/7", calls[0])

    def test_list_reviews_paginates_and_maps_the_fields(self) -> None:
        page = json.dumps(
            [
                {
                    "user": {"login": "dpologdvinity"},
                    "state": "APPROVED",
                    "commit_id": HEAD,
                    "author_association": "OWNER",
                },
                {"user": None, "state": "COMMENTED", "commit_id": HEAD, "author_association": "NONE"},
            ]
        )
        calls: list[list[str]] = []

        def runner(argv: Sequence[str], stdin: str | None) -> str:
            calls.append(list(argv))
            return page

        reviews = gh.GhCli("o/r", runner).list_reviews(7)
        self.assertEqual(reviews[0], approval())
        self.assertEqual(reviews[1].user, "")  # a deleted account has no login
        self.assertIn("--paginate", calls[0])
        self.assertIn("repos/o/r/pulls/7/reviews?per_page=100", calls[0])


if __name__ == "__main__":
    unittest.main()
