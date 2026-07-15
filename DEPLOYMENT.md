# Deploying MindCare for free (Hugging Face Spaces)

This gets MindCare live on **Hugging Face Spaces** for your testers, for
$0. Spaces' free CPU Basic tier gives you **2 vCPU / 16GB RAM**, which
comfortably fits DistilBERT + Qwen2.5-1.5B running together on CPU, plus a
real shareable HTTPS URL and no server management.

There's one real trade-off to know about upfront (Section 5 explains it and
how to work around it if it matters for you): **free Spaces have ephemeral
storage**, so the SQLite conversation log resets whenever the Space sleeps
and wakes back up. The model weights don't have this problem -- the
`Dockerfile` in this repo bakes them into the image at build time
specifically to avoid it.

---

## 1. Train the classifier locally first (recommended)

```bash
pip install -r requirements.txt
python train_distilbert.py --epochs 8 --batch-size 16
```

This creates `models/distilbert-intent/`. Committing this to your Space
means testers get accurate intent classification from the start rather than
the untrained fallback (`chatbot.py` will run without it, just poorly).

---

## 2. Create the Space

1. Sign up / log in at [huggingface.co](https://huggingface.co).
2. Click your profile -> **New Space**.
3. Pick a name (e.g. `mindcare`), set visibility (Public is fine for a
   tester group, or Private if you want to control access via HF accounts).
4. **SDK: Docker**. Template: "Blank."
5. Create the Space. HF gives you a git remote URL like
   `https://huggingface.co/spaces/<your-username>/mindcare`.

---

## 3. Prepare the repo for the Space

Spaces read their configuration from YAML frontmatter at the top of the
repo's root `README.md`. This project ships a ready-made one at
`SPACE_README.md` -- use it as your Space's `README.md`:

```bash
cd MindCare
cp README.md README.full.md      # keep your full project README around
cp SPACE_README.md README.md     # this one has the required frontmatter
```

Also, **the trained classifier checkpoint needs to actually be in this push**,
even though it's a large binary and your general `.gitignore` normally
excludes `models/`. Hugging Face repos use Git LFS automatically for large
files, so this is exactly what Spaces is built for -- force-add it for this
push:

```bash
git init                      # if you haven't already, for this repo
git add .
git add -f models/            # override .gitignore just for this directory
git commit -m "Initial MindCare Space"
```

---

## 4. Push

```bash
git remote add space https://huggingface.co/spaces/<your-username>/mindcare
git push space main
```

You'll be prompted for HF credentials -- use a
[Hugging Face access token](https://huggingface.co/settings/tokens) as the
password if prompted (not your account password).

Building takes a while the first time (10-20 minutes) because the Dockerfile
downloads and bakes in the Qwen weights during the build step. Watch progress
on the **Logs** tab of your Space page. Once it says "Running," your app is
live at:

```
https://<your-username>-mindcare.hf.space
```

That's the URL you share with your testers.

---

## 5. The ephemeral storage trade-off (and what to do about it)

Free Spaces don't include a persistent disk. Two different things are
affected by this differently:

- **Model weights**: not affected. They're baked into the Docker image at
  build time (see the Dockerfile), so they survive every sleep/wake/restart
  with zero re-download.
- **`mindcare.db` (conversation logs)**: *is* affected. It lives on the
  container's runtime filesystem, which resets when the Space sleeps
  (typically after ~48 hours of inactivity) and wakes back up, or whenever
  you push a new commit.

For a few-weeks tester group, here's the practical read:
- If you mainly care about **testers being able to use the app reliably**
  and don't need every single logged conversation preserved permanently,
  this is a non-issue -- just be aware a redeploy or a long quiet spell
  clears the log, and periodically export `mindcare.db` via the Space's
  file browser if you want to keep something for your FYP writeup.
- If you need **continuous, guaranteed persistence** across sleep cycles
  (e.g. you're actively analyzing logs day-to-day), the zero-cost fix is a
  free-tier managed Postgres (Supabase or Neon both have free tiers) instead
  of SQLite -- this would mean adapting `conversation_logger.py` to use
  `psycopg2`/`sqlalchemy` against a `DATABASE_URL` env var instead of a local
  file. This is a real code change beyond what's in this repo today; happy
  to build it out if you decide you need it.

---

## 6. Keeping it awake / checking on it

- Free Spaces sleep after a period of no traffic and wake on the next visit
  (with a cold start of maybe 10-30 seconds, since weights are already
  baked in rather than downloaded). Tell your testers a first message after
  a quiet stretch might take a little longer.
- **Logs**: Space page -> **Logs** tab. Watch for repeated
  `[chatbot] LIME explanation failed` or crash loops.
- **Crisis tier events**: `app.py` logs a warning every time any crisis tier
  fires -- worth periodically checking these look right.
- To update the app later, just commit and `git push space main` again;
  Spaces rebuilds automatically.

---

## 7. If you outgrow the free tier

If testers report Qwen replies feel slow, or you want the SQLite/persistence
issue solved without extra code, Spaces also lets you upgrade *just the
hardware* from the Settings tab without changing any code -- no redeploy of
your own, HF handles moving it. Reasonable next steps at low cost:
`CPU Upgrade` tier for more RAM/CPU, or a small GPU tier if generation speed
becomes the bottleneck.

`render.yaml` is also still included in this repo as a paid alternative
(Render.com, ~$25/mo Standard plan) if you ever want a platform with a true
persistent disk out of the box instead of the Postgres workaround above --
not needed for a free deployment, just left in for later.
