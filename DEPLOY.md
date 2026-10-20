# Deploy guide

Three steps: train the model in Colab, put the code on GitHub, launch the demo on
Hugging Face Spaces. Everything is free. Total time: about 30 minutes.

---

## Step 1: Train the model (Colab, about 5 minutes)

1. Go to https://colab.research.google.com → **File → Upload notebook** →
   choose `notebooks/fashion_mnist_experiments.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.** The last cell downloads `demo_files.zip`.
4. Unzip it. You get:
   - `fashion_cnn.keras` → move it into the repo's `app/` folder
   - `examples/` (10 PNGs) → move the PNGs into `app/examples/`
   - `results.csv` → optional; if your numbers differ, update the README table
5. **File → Download → Download .ipynb** and replace
   `notebooks/fashion_mnist_experiments.ipynb` with it, so the notebook on GitHub shows
   its outputs (curves, tables, accuracies).

## Step 2: GitHub repo

**Option A, website (no git needed):**
1. https://github.com/new → name `fashion-mnist-classifier` → Public →
   do **not** add a README (you already have one) → **Create repository**.
2. Click **uploading an existing file**, drag in everything inside the
   `fashion-mnist-classifier` folder (including `app/`, `src/`, `notebooks/`, `docs/`,
   `results/`), then **Commit changes**.

**Option B, git:**
```bash
cd fashion-mnist-classifier
git init
git add .
git commit -m "Fashion-MNIST classifier: experiments, training scripts and demo"
git branch -M main
git remote add origin https://github.com/kunalv14/fashion-mnist-classifier.git
git push -u origin main
```

Then on the repo page: click the ⚙ next to **About** → add a description
("CNN that classifies clothing images with 91.5% accuracy"), the demo link, and topics
`deep-learning`, `tensorflow`, `keras`, `cnn`, `computer-vision`, `gradio`.

## Step 3: Live demo on Hugging Face Spaces

1. Sign up at https://huggingface.co/join (free).
2. https://huggingface.co/new-space → name `fashion-mnist-classifier` →
   SDK **Gradio** → hardware **CPU basic (free)** → Public → **Create Space**.
3. On the Space page: **Files → Add file → Upload files**, upload the **contents** of the
   `app/` folder so they sit at the top level of the Space:
   ```
   README.md          (the one inside app/ — it has the Space settings at the top)
   app.py
   requirements.txt
   fashion_cnn.keras
   examples/          (the 10 PNGs)
   ```
   Commit. The Space builds for 2–4 minutes, then shows the app.
4. Copy the Space URL (`https://huggingface.co/spaces/kverma19/fashion-mnist-classifier`)
   into the GitHub README's "Live demo" link and your resume.

### If the build fails
- **Open the Logs tab** on the Space; the last lines name the problem.
- *Gradio version error:* in `README.md` on the Space, change `sdk_version` to the
  newest version listed at https://pypi.org/project/gradio/ and commit.
- *Model fails to load (Keras version):* in `requirements.txt`, set `tensorflow-cpu` to
  the version Colab printed in the notebook's first cell, e.g. `tensorflow-cpu==2.19.0`.
- Free Spaces sleep after about 48 hours without visitors; the first visit after that
  takes a minute to wake up. That's normal.
