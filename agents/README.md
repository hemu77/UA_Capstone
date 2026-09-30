# Team Reviewer Agent Package

The team can reuse the project's **read-only research architecture reviewer**.
It is a checklist and task prompt for an assistant, not a background service,
installed model, human reviewer or authority to spend API money.

## Files

- [Review task prompt](review-architecture.md): portable instructions to use in a
  coding-assistant review session or follow manually.
- [Canonical project skill](../.agents/skills/review-architecture/SKILL.md): the
  existing repository-specific skill used during the revision. It is included in Git.
- [Team approval](../docs/TEAM_APPROVAL.md): scientific/budget decisions the agent
  must not fill in or approve on anyone's behalf.

## How to Use It

Open a coding-assistant session on this repository, provide the bounded files or
commit to review, and ask it to follow `agents/review-architecture.md` and the
canonical skill. If your tool discovers `.agents/skills`, select
`review-architecture`. Discovery depends on the tool; cloning does not launch it.
No OpenAI research API key is needed for offline repository review; the assistant
itself may have its own subscription/token cost. RTK is required for automated
shell commands by this checklist; if unavailable, report that instead of silently
installing software or switching to paid work.

Example request:

> Read agents/review-architecture.md and the canonical skill. Review only
> paid_study.py, revision224.py, the recovery tests and current calibration
> report. Check retained charges, duplicate-payment prevention, source identity
> and separation of calibration/main approval. Do not edit files, call an API,
> change any gate, or spawn other agents. Return evidence-backed findings and
> the exact checks actually run.

Use one bounded reviewer by default, with explicit permission where required.
Do not ask for an unlimited loop until it declares "100% correct." A clean review
means no material issue was found within its stated scope, not proof of no bugs.
The execution owner verifies findings before fixing anything.
