# Workflow Walkthrough — Finishing One Clip, Start to Finish

This is a **hands-on, step-by-step walkthrough**. Follow it once and you will have reviewed a complete project archive from the moment you load the `.zip` to the moment you export the finished `.json`.

It is deliberately linear: each step tells you what to do, and a **checkpoint** tells you how to know it worked before you move on.

> 📌 **Companion document:** this walkthrough is the tutorial. For a feature-by-feature reference (what every control does, the full troubleshooting table, and a glossary), see `docs/USER_GUIDE.md`.

---

## What you will accomplish

By the end of this walkthrough you will have:

- Loaded one project `.zip` into the reviewer.
- Reviewed **every** tracklet in the clip — confirming or correcting its taxonomic label and giving its mask a verdict.
- Exported `<clip>.review.json` and `<clip>.review.csv` containing your complete review.

**Running example:** throughout this guide we review a fictional clip called **`clip_007.zip`**, which contains **123 frames** and **24 tracklets**. Substitute your own clip's numbers as you go.

**Suggested time:** roughly 1–3 minutes per tracklet, so a 24-tracklet clip takes around 30 minutes. The keyboard shortcuts in [Step 3](#step-3--review-a-tracklet-the-core-loop) make it much faster.

---

## Before you begin — prerequisites

Confirm all three of these before you start. Skipping this is the most common cause of a wasted session.

| #   | Requirement                                    | How to check                                                                                                                                 |
| --- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **The mask service is running**                | The helper service must be up before you open the app, or no mask overlays will appear. See [Step 0](#step-0--start-the-app).                |
| 2   | **You have the project `.zip`**                | One `.zip` per clip, containing a `frames/` folder and an `annotation.json` at its root.                                                     |
| 3   | **The `.zip` is DEFLATE or STORED compressed** | If it was bundled with LZMA or BZIP2 the browser cannot read it and the app will refuse to open it. Re-export with `--compression deflated`. |

> ⚠️ **Do not rename or re-zip the archive yourself** unless you know what you are doing. The frame image names inside must match the annotation JSON exactly.

---

## Step 0 — Start the app

The reviewer has two parts: a small helper service that decodes masks, and the browser interface itself.

1. **Start the mask service** (once per session) and leave it running in its own terminal.
2. **Start the web interface** and note the address it prints (typically `http://localhost:5173`).
3. **Open that address in your browser.**

> 🖼️ **IMAGE NEEDED — `walkthrough-01-terminal-running.png`**
> _What to capture:_ A screenshot of the two terminal windows side by side (or one terminal with both commands), showing the mask service reporting it is running and the web server printing its local URL. Nothing sensitive visible.
> _Purpose:_ Reassures the reader that "start the app" means two processes, and shows what a healthy startup looks like.

**✅ Checkpoint:** the browser shows the **upload screen** — a large dashed drop zone reading _"Drop a project `.zip` archive here, or click to browse."_ If you instead see errors in the terminal, fix those before continuing.

---

## Step 1 — Load the `.zip`

![PLACEHOLDER — walkthrough-02-upload-screen](./images/walkthrough-02-upload-screen.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-02-upload-screen.png`**
> _What to capture:_ The upload screen with a `.zip` file being dragged over the drop zone (the zone highlighted in its "drag over" state). A cursor holding a `.zip` file is ideal.
> _Purpose:_ Makes the drag-and-drop action unambiguous for first-time users.

1. Open your file manager and locate **`clip_007.zip`**.
2. **Drag the file onto the drop zone** and release.
   — _or_ —
   **Click the drop zone**, browse to the file, and select it.
3. A **"Loading"** screen appears while the app reads the frames and annotation JSON. Larger clips take a few seconds.

**✅ Checkpoint:** after loading, the review workspace appears with headline **`clip_007`** in the top-left corner. If instead you land on **"Could not open archive"**, jump to [If something goes wrong](#if-something-goes-wrong).

---

## Step 2 — Orientation: confirm the clip loaded correctly

Before reviewing anything, take ten seconds to confirm the app read your file the way you expect.

![PLACEHOLDER — walkthrough-03-header-check](./images/walkthrough-03-header-check.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-03-header-check.png`**
> _What to capture:_ A close-up of the header bar only, showing the clip name and the metadata row — `123 frames`, `25 fps`, `1920×1080`, `24 tracklets` — and the progress bar reading `0 / 24 verified`.
> _Purpose:_ Teaches the reader where to verify the clip loaded with the right dimensions and tracklet count.

Look at the **header bar** and confirm:

| What to look for   | In our example    | Why it matters                                             |
| ------------------ | ----------------- | ---------------------------------------------------------- |
| **Clip name**      | `clip_007`        | Confirms you opened the right file.                        |
| **Frame count**    | `123 frames`      | Sanity-check against the clip you expected.                |
| **FPS**            | `25 fps`          | Playback speed will follow this.                           |
| **Resolution**     | `1920×1080`       | The canvas will fit this to the window.                    |
| **Tracklet count** | `24 tracklets`    | This is how many objects you must review.                  |
| **Progress**       | `0 / 24 verified` | Confirms you are starting fresh (or resumed, if non-zero). |

Also glance for an amber warning such as **"N frame(s) missing from archive"**. If present, the clip is incomplete — you can still review it, but note it for your data provider.

**✅ Checkpoint:** the numbers match your expectations and the progress bar reads `0 / 24 verified` (or your own resume point). You are ready to review.

---

## Step 3 — Review a tracklet (the core loop)

This is the loop you will repeat for all 24 tracklets. Learn it once here in full, then it is quick.

### 3a. Select a tracklet

Look at the **tracklet list** on the right. The first tracklet is already selected for you, and the playhead has jumped to the **first frame where that object appears**.

![PLACEHOLDER — walkthrough-04-first-tracklet-selected](./images/walkthrough-04-first-tracklet-selected.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-04-first-tracklet-selected.png`**
> _What to capture:_ A screenshot showing the tracklet list with the first row highlighted as selected, and the Inspector below showing that tracklet's details. An arrow or callout can point from the list row to the Inspector.
> _Purpose:_ Establishes the link between "a row in the list" and "the panel you edit below it".

**✅ Checkpoint:** exactly one row in the list is highlighted, and the Inspector is populated (it no longer says _"Select a tracklet to review its label and mask."_).

### 3b. Inspect the mask

Play the clip and watch the **coloured overlay** on the video.

1. Press **`Space`** to play and **`Space`** again to pause.
2. Press **`←`** and **`→`** to step frame by frame through the object's span.
3. Drag the **Overlay** slider down to ~30% to check the mask still sits on the animal, then back up to see its outline clearly.

![PLACEHOLDER — walkthrough-05-mask-inspection](./images/walkthrough-05-mask-inspection.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-05-mask-inspection.png`**
> _What to capture:_ The video canvas mid-clip with a clearly drawn coloured mask overlay, and the control strip beneath it visible (Play button, step buttons, slider, timecode, Overlay slider). Overlay the frame number on the image if helpful.
> _Purpose:_ Shows the reader the practical act of inspecting an overlay against the animal.

Ask yourself: **does the mask tightly and consistently outline the animal on every frame it appears?** Decide:

- ✅ **Accurate** — it fits throughout.
- ❌ **Inaccurate** — it is too large, too small, drifting, or covering the wrong thing on some frames.
- ❓ **Unsure** — you cannot tell (occlusion, poor visibility, ambiguous edges).

**✅ Checkpoint:** you have formed a clear verdict for this tracklet's mask. If a particular frame is the problem, note its number — you will put it in the comment later.

### 3c. Verify the taxonomic label

Now look at the **Inspector** below the list. The pipeline's predicted taxonomy is pre-filled. Confirm or correct it.

![PLACEHOLDER — walkthrough-06-taxonomy-autocomplete](./images/walkthrough-06-taxonomy-autocomplete.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-06-taxonomy-autocomplete.png`**
> _What to capture:_ The Inspector's "Taxonomic label" section with a rank field mid-typing and the autocomplete dropdown open showing 2–3 suggestions with ranks, plus the Confirm label button below.
> _Purpose:_ Demonstrates the one genuinely interactive editing feature in the loop.

1. Read the **Kingdom → Species** fields and the **Common name**.
2. To correct a rank, **click into that field and type** the correct name. After a brief pause a **dropdown of matching taxa** appears.
3. Use **`↑` / `↓`** and **`Enter`**, or click, to accept a suggestion. The tool then fills in that rank **and every broader rank above it** (choose a species and its genus, family, order… all fill in).
4. Fill in the **Common name** manually if needed.
5. When satisfied, click **Confirm label**. A **✓ confirmed** badge appears.

> ⚠️ **If you change any field after confirming, the confirmation clears** — the button returns to "Confirm label" and you must confirm again. This protects against half-edited labels slipping through.

> ⚠️ **Do not confirm a label you are not confident about.** Leave it unconfirmed so the tracklet stays visible under the **In progress** filter.

**✅ Checkpoint:** the **✓ confirmed** badge is showing next to the label button.

### 3d. Record the mask verdict and comment

Still in the Inspector, under **Mask quality**:

1. Click the verdict that matches your decision from 3b: **Accurate**, **Inaccurate**, or **Unsure** — or just press **`3`** / **`4`** for Accurate / Inaccurate.
2. Optionally type a short **comment**, e.g. _"mask drifts off the tail after frame 40"_.

![PLACEHOLDER — walkthrough-07-verdict-comment](./images/walkthrough-07-verdict-comment.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-07-verdict-comment.png`**
> _What to capture:_ The Inspector's "Mask quality" section with the **Inaccurate** button highlighted (so it differs from the USER_GUIDE image, which shows Accurate) and a realistic comment typed in the box.
> _Purpose:_ Shows a verdict selected _and_ a comment filled, i.e. a completed tracklet.

**✅ Checkpoint — this is the important one:** the tracklet's row in the list now reads **Verified**. That only happens when the label is confirmed **and** a mask verdict is set. If it still says **In progress**, you have done one half of the pair — check both 3c and 3d.

> 💡 Your work is saved automatically in the browser. You can close the tab and resume later by re-opening the same `.zip`.

---

## Step 4 — Repeat for every remaining tracklet

Now work through the rest of the clip. The fastest way is the dedicated shortcut:

1. Press **`n`** to **jump to the next unverified tracklet**. The list scrolls to it and the playhead jumps to its first visible frame.
2. Run the loop from [Step 3](#step-3--review-a-tracklet-the-core-loop): inspect the mask → confirm/correct the label → set the verdict.
3. Press **`n`** again. Repeat.

The progress bar in the header updates as you go, e.g. `7 / 24 verified`.

![PLACEHOLDER — walkthrough-08-tracklet-list-progress](./images/walkthrough-08-tracklet-list-progress.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-08-tracklet-list-progress.png`**
> _What to capture:_ The tracklet list showing a mix of statuses — several **Verified**, one or two **In progress**, several **Pending** — with the **In progress** filter button selected. Capture the progress bar above it too if possible.
> _Purpose:_ Shows a realistic mid-review state and demonstrates the status filters.

**Useful while working through the clip:**

- Every tracklet's mask is drawn at once, so you can see when two objects' masks are being confused. The selected object's mask is drawn in **white**.
- If you need to re-check one later, use the **search box** to filter by label or ID.

**✅ Checkpoint:** you have visited every tracklet. Tracklets you deliberately left as **In progress** are fine as long as you know why — just don't mistake them for finished ones.

---

## Step 5 — Confirm the clip is fully reviewed

Before exporting, verify the clip is genuinely complete.

1. Look at the **progress bar** in the header — it should read **`24 / 24 verified`** (i.e. numerator equal to the tracklet count).
2. Optionally click the **Pending** filter and the **In progress** filter. A fully reviewed clip shows _"No tracklets match."_ for both.

![PLACEHOLDER — walkthrough-09-progress-complete](./images/walkthrough-09-progress-complete.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-09-progress-complete.png`**
> _What to capture:_ A close-up of the header showing a **full** progress bar reading `24 / 24 verified`, with the Export button visible.
> _Purpose:_ Defines the completion signal the reader is aiming for.

> 💡 **You do not have to reach 24 / 24.** If some tracklets genuinely cannot be judged, leaving them **In progress** or giving them an **Unsure** verdict is a legitimate outcome — just make sure that is a deliberate choice, not an oversight.

**✅ Checkpoint:** every tracklet is either **Verified** or a deliberate, known exception.

---

## Step 6 — Export the review

Click **Export** in the header.

![PLACEHOLDER — walkthrough-10-export-button](./images/walkthrough-10-export-button.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-10-export-button.png`**
> _What to capture:_ A close-up of the header with a cursor about to click the **Export** button, or the button circled.
> _Purpose:_ Pinpoints the exact control that produces the deliverable.

Your browser downloads **two files** (they may appear in your Downloads folder or a "Save as" dialog, depending on browser settings):

| File                       | Contents                                                                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`clip_007.review.json`** | The complete structured record: clip metadata plus, for every tracklet, the original pipeline values, your final values, the label confirmation flag, the mask verdict, and your comment. |
| **`clip_007.review.csv`**  | The same content as a flat table, ready to open in Excel, R, or Python.                                                                                                                   |

> 📌 Both files contain the **original** prediction and your **final** decision **side by side**, so a downstream script can diff what the model produced against what you concluded.

**✅ Checkpoint:** two files named after your clip (`.review.json` and `.review.csv`) are in your downloads.

---

## Step 7 — Verify the exported JSON

Do not assume the export is good — spend thirty seconds checking it.

1. Open **`clip_007.review.json`** in a text editor.
2. Confirm it is valid JSON and that the top-level fields look right:

| Field                    | What you should see                                                  |
| ------------------------ | -------------------------------------------------------------------- |
| `clip`                   | `"clip_007"`                                                         |
| `exportedAt`             | Today's date/time (an ISO timestamp)                                 |
| `fps`, `width`, `height` | The clip's values, e.g. `25`, `1920`, `1080`                         |
| `reviews`                | An array with **one entry per tracklet** (24 entries in our example) |
| `csv`                    | The same table embedded as a string                                  |

3. Spot-check one entry you remember reviewing. For that tracklet you should see:

- `label_confirmed` — `"true"` for a tracklet you confirmed.
- `mask_verdict` — `"good"`, `"bad"`, or `"unsure"`.
- `final_*` fields — either your corrected values, or the originals if you changed nothing.
- `comment` — your note, if you wrote one.

![PLACEHOLDER — walkthrough-11-export-json-inspected](./images/walkthrough-11-export-json-inspected.png)

> 🖼️ **IMAGE NEEDED — `walkthrough-11-export-json-inspected.png`**
> _What to capture:_ A text editor showing the exported `.review.json` open, with the top-level fields (`clip`, `exportedAt`, `fps`, `width`, `height`) and the start of the `reviews` array visible. Blur or use a synthetic clip if your data is sensitive.
> _Purpose:_ Shows the reader what a correct export actually looks like, so they can self-check.

**✅ Checkpoint:** the JSON opens cleanly, has one `reviews` entry per tracklet, and shows your verdicts. **That clip is now finished.**

---

## Step 8 — Move on to the next clip

1. Click **Open another** in the header.
2. You return to the upload screen.
3. Drop the next project `.zip` and repeat from [Step 1](#step-1--load-the-zip).

Each clip keeps its own separate review state, so switching back and forth is safe as long as you use the same browser on the same machine.

> ⚠️ **Remember:** review state lives in **this browser on this machine only**. It does not travel with the file and is lost if you clear browser data. **Export each clip as you finish it** — the exported files are the durable record.

---

## Quick reference card

Keep this beside you while reviewing.

**The loop**

| Action                           | How                                                  |
| -------------------------------- | ---------------------------------------------------- |
| Jump to next unverified tracklet | **`n`**                                              |
| Play / pause                     | **`Space`**                                          |
| Step one frame back / forward    | **`←`** / **`→`**                                    |
| Mask is Accurate                 | **`3`**                                              |
| Mask is Inaccurate               | **`4`**                                              |
| Mask is Unsure                   | Click **Unsure** (no shortcut)                       |
| Toggle all masks                 | **`x`**                                              |
| Finish the clip                  | Progress reads **N / N verified** → click **Export** |

**Done means:** label **✓ confirmed** **and** a mask verdict set → the row shows **Verified**.

**Deliverable:** `<clip>.review.json` + `<clip>.review.csv`.

---

## If something goes wrong

| Symptom                                                            | Likely cause                                        | Fix                                                                                    |
| ------------------------------------------------------------------ | --------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **"Could not open archive"** mentioning compression or _method 14_ | The `.zip` uses LZMA/BZIP2.                         | Re-export with `--compression deflated`, or ask your data provider for a DEFLATE copy. |
| **"Could not open archive"** with another message                  | Missing `frames/` folder or annotation JSON.        | Verify the archive's internal layout matches the required structure.                   |
| **No coloured overlay on the video**                               | Mask service is not running.                        | Start the helper service (Step 0), then re-select the tracklet.                        |
| **Grey "Frame unavailable" box**                                   | Frames are missing from the archive.                | Review the available frames; notify your data provider.                                |
| **Shortcuts stopped working**                                      | Focus is in a text field.                           | Click the video canvas or press `Escape`.                                              |
| **Progress empty after reopening**                                 | Different browser/machine, or browser data cleared. | Redo the review and export promptly next time.                                         |

For a fuller troubleshooting list and a glossary of terms, see `docs/USER_GUIDE.md`.
