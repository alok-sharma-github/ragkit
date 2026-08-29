# Working in this repo

## How to end a reply

Whenever you have actually done something in response to a prompt — read the
code, run a command, change a file, investigate a question — end that reply with
the report below. **Every prompt you work on gets one**, not just the last prompt
of a session. Skip it only for pure back-and-forth where you did no work at all:
answering a question about the conversation itself, or asking a clarifying
question before starting.

Write it for someone who has not read any code and was not watching you work.
Describe behaviour, not filenames. Use no project jargon unless it is defined in
`docs/GLOSSARY.md`.

Put it at the very end, after everything else, using exactly these headings and
this order:

```
## What changed
2–4 bullets. What the system can do now that it couldn't before. Describe
behaviour, not files. If nothing changed because the work was investigation,
say what is now known instead — and say plainly that nothing was modified.

## How it works now
3–5 sentences. Explain the mechanism to a competent engineer who has never
seen this repo.

## What I decided on my own
Anything you chose without being told. Flag anything worth reversing if the
guess was wrong. Say "nothing" if every choice was specified.

## What I'm not sure about
Things that might be broken, untested, or fragile. Say "none" only if you
mean it.

## Your call before I continue
1–3 specific questions, each with options. Do not proceed past these.
```
