# ColdRoot v3

ColdRoot v3 uses `Smogy/SMOGY-Ai-images-detector` instead of the previous CIFAKE-based detector. Its model card reports strong held-out performance and gives out-of-domain results for DALL-E, Flux, Imagen and Stable Diffusion; real-world accuracy can still vary.

The detector also analyzes three views of each image (original, mirrored, center crop) and uses the median score to reduce sensitivity to one view.

Run on Windows:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000`.
