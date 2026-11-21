"""Unit tests for tools/ci/fix_proof_waiver.py and the label-event and permission calls of gh.GhCli (no network).

Run with: python -m unittest discover -s tools/ci -p 'test_*.py'
They pin who can waive fix-proof: the `no-test-needed` label counts only when it is on the pull request now and its
latest `labeled` event was made by a person (not a bot) with write, maintain or admin access. Anything else, a failed
API call included, is no waiver.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

import fix_proof_waiver
import gh
from fakes import FakeGitHub
from gh import LabelEvent

LABEL = "no-test-needed"
HEAD = "a" * 40


def labelled_pr(fake: FakeGitHub, *events: LabelEvent, labels: Sequence[str] = (LABEL,)) -> int:
    """An open pull request carrying `labels` now, whose label history is `events` (oldest first)."""
    number = fake.add_pull_request(HEAD, author="dpologdvinity", labels=list(labels))
    fake.label_history[number] = list(events)
    return number


def labeled(actor: str, *, label: str = LABEL, actor_type: str = "User") -> LabelEvent:
    """A `labeled` event for `label` made by `actor`."""
    return LabelEvent(event="labeled", label=label, actor=actor, actor_type=actor_type)


def unlabeled(actor: str, *, label: str = LABEL) -> LabelEvent:
    """An `unlabeled` event for `label` made by `actor`."""
    return LabelEvent(event="unlabeled", label=label, actor=actor, actor_type="User")


def with_permissions(**logins: str) -> FakeGitHub:
    """A FakeGitHub where each keyword login has that repository permission (the owner is admin)."""
    fake = FakeGitHub()
    fake.permissions.update({"dpologdvinity": "admin", **logins})
    return fake


class EvaluateTests(unittest.TestCase):
    def test_owner_applied_label_waives(self) -> None:
        fake = with_permissions()
        number = labelled_pr(fake, labeled("dpologdvinity"))
        verdict = fix_proof_waiver.evaluate(fake, number, LABEL)
        self.assertTrue(verdict.waived)
        self.assertIn("dpologdvinity", verdict.reason)

    def test_write_and_maintain_access_waive(self) -> None:
        for permission in ("write", "maintain", "admin"):
            with self.subTest(permission=permission):
                fake = with_permissions(helper=permission)
                number = labelled_pr(fake, labeled("helper"))
                self.assertTrue(fix_proof_waiver.evaluate(fake, number, LABEL).waived)

    def test_label_applied_by_the_agent_does_not_waive(self) -> None:
        # `gh pr create --label no-test-needed` from a Claude session: the actor is the Claude app.
        fake = with_permissions()
        number = labelled_pr(fake, labeled("claude[bot]", actor_type="Bot"))
        verdict = fix_proof_waiver.evaluate(fake, number, LABEL)
        self.assertFalse(verdict.waived)
        self.assertIn("bot", verdict.reason)

    def test_a_bot_never_waives_whatever_its_name_or_access(self) -> None:
        cases = [("claude[bot]", "User"), ("renovate", "Bot"), ("github-actions[bot]", "Bot")]
        for actor, actor_type in cases:
            with self.subTest(actor=actor, actor_type=actor_type):
                fake = with_permissions(**{actor: "admin"})
                number = labelled_pr(fake, labeled(actor, actor_type=actor_type))
                self.assertFalse(fix_proof_waiver.evaluate(fake, number, LABEL).waived)

    def test_read_or_triage_access_does_not_waive(self) -> None:
        for permission in ("read", "triage", "none", ""):
            with self.subTest(permission=permission):
                fake = with_permissions(stranger=permission)
                number = labelled_pr(fake, labeled("stranger"))
                self.assertFalse(fix_proof_waiver.evaluate(fake, number, LABEL).waived)

    def test_the_latest_labeled_event_decides(self) -> None:
        fake = with_permissions()
        # The owner's label was removed and the agent put it back: the agent's application is the one that stands.
        number = labelled_pr(
            fake, labeled("dpologdvinity"), unlabeled("dpologdvinity"), labeled("claude[bot]", actor_type="Bot")
        )
        self.assertFalse(fix_proof_waiver.evaluate(fake, number, LABEL).waived)
        # The owner re-applying it after the agent makes it count.
        fake.label_history[number].append(labeled("dpologdvinity"))
        self.assertTrue(fix_proof_waiver.evaluate(fake, number, LABEL).waived)

    def test_events_for_other_labels_are_ignored(self) -> None:
        fake = with_permissions()
        number = labelled_pr(fake, labeled("claude[bot]", actor_type="Bot"), labeled("dpologdvinity", label="agent"))
        self.assertFalse(fix_proof_waiver.evaluate(fake, number, LABEL).waived)

    def test_label_not_on_the_pull_request_does_not_waive(self) -> None:
        fake = with_permissions()
        number = labelled_pr(fake, labeled("dpologdvinity"), labels=("fix",))
        verdict = fix_proof_waiver.evaluate(fake, number, LABEL)
        self.assertFalse(verdict.waived)
        self.assertIn("does not carry", verdict.reason)

    def test_label_without_a_labeled_event_does_not_waive(self) -> None:
        # The event list can lag behind the label: no recorded application means no waiver (re-run the check).
        fake = with_permissions()
        number = labelled_pr(fake)
        self.assertFalse(fix_proof_waiver.evaluate(fake, number, LABEL).waived)

    def test_unknown_permission_lookup_fails_closed(self) -> None:
        fake = FakeGitHub()  # no permission on record: the fake raises GhError
        number = labelled_pr(fake, labeled("dpologdvinity"))
        verdict = fix_proof_waiver.evaluate(fake, number, LABEL)
        self.assertFalse(verdict.waived)
        self.assertIn("could not", verdict.reason)

    def test_event_listing_failure_fails_closed(self) -> None:
        fake = with_permissions()
        number = labelled_pr(fake, labeled("dpologdvinity"))
        fake.fail_label_events = True
        self.assertFalse(fix_proof_waiver.evaluate(fake, number, LABEL).waived)


class MainTests(unittest.TestCase):
    def run_main(self, fake: FakeGitHub, number: int, summary: Path | None = None) -> str:
        argv = ["--pr", str(number)]
        if summary is not None:
            argv += ["--summary", str(summary)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(fix_proof_waiver.main(argv, gh=fake, repo="o/r"), 0)
        return out.getvalue()

    def test_main_prints_the_output_line_and_writes_the_reason(self) -> None:
        fake = with_permissions()
        number = labelled_pr(fake, labeled("dpologdvinity"))
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.md"
            self.assertEqual(self.run_main(fake, number, summary), "waived=true\n")
            self.assertIn("dpologdvinity", summary.read_text(encoding="utf-8"))

    def test_main_reports_no_waiver_with_exit_zero(self) -> None:
        fake = with_permissions()
        number = labelled_pr(fake, labeled("claude[bot]", actor_type="Bot"))
        self.assertEqual(self.run_main(fake, number), "waived=false\n")

    def test_default_label_is_no_test_needed(self) -> None:
        self.assertEqual(fix_proof_waiver.WAIVER_LABEL, LABEL)


class GhCliTests(unittest.TestCase):
    def test_label_events_reads_every_page_and_keeps_label_events_only(self) -> None:
        pages = json.dumps(
            [
                {"event": "labeled", "actor": {"login": "dpologdvinity", "type": "User"}, "label": {"name": LABEL}},
                {"event": "closed", "actor": {"login": "dpologdvinity", "type": "User"}},
            ]
        ) + json.dumps(
            [
                {"event": "unlabeled", "actor": {"login": "claude[bot]", "type": "Bot"}, "label": {"name": LABEL}},
                {"event": "labeled", "actor": None, "label": {"name": "x"}},
            ]
        )
        calls: list[list[str]] = []

        def runner(argv: Sequence[str], stdin: str | None) -> str:
            calls.append(list(argv))
            return pages

        events = gh.GhCli("o/r", runner).label_events(7)
        self.assertEqual(calls, [["gh", "api", "--paginate", "repos/o/r/issues/7/events?per_page=100"]])
        self.assertEqual(
            events,
            [
                LabelEvent("labeled", LABEL, "dpologdvinity", "User"),
                LabelEvent("unlabeled", LABEL, "claude[bot]", "Bot"),
                LabelEvent("labeled", "x", "", ""),
            ],
        )

    def test_permission_reads_the_permission_field_and_quotes_the_login(self) -> None:
        # `permission` is admin, write, read or none (maintain reads as write, triage as read); `role_name` can be a
        # custom role's name, so it is not used.
        calls: list[list[str]] = []

        def runner(argv: Sequence[str], stdin: str | None) -> str:
            calls.append(list(argv))
            return json.dumps({"permission": "write", "role_name": "custom-releaser"})

        self.assertEqual(gh.GhCli("o/r", runner).permission("claude[bot]"), "write")
        self.assertEqual(calls, [["gh", "api", "repos/o/r/collaborators/claude%5Bbot%5D/permission"]])

    def test_permission_without_the_field_is_none(self) -> None:
        def runner(argv: Sequence[str], stdin: str | None) -> str:
            return json.dumps({"message": "odd"})

        self.assertEqual(gh.GhCli("o/r", runner).permission("someone"), "none")


if __name__ == "__main__":
    unittest.main()
