# RCC Tumor Classification Project

This project classifies kidney tumor subtype from biopsy/WSI image patches using multiple deep learning models.

It is designed to run on **Windows + Ubuntu WSL + VS Code (WSL mode)**.

---

## 1. Recommended Environment

Use:

- **Windows**
- **Ubuntu through WSL 2**
- **VS Code with the WSL extension**

### Install WSL Ubuntu
Open **PowerShell as Administrator**:

```powershell
wsl --install
```

Restart if needed.

### Check WSL version
```powershell
wsl -l -v
```

Ubuntu should show:

```text
VERSION 2
```

### Open the project in WSL
Inside Ubuntu:

```bash
cd ~
git clone <repository-url>
cd Grad_Project
code .
```

Make sure VS Code shows:

```text
WSL: Ubuntu
```

---

## 2. Install Basic Tools

Inside Ubuntu:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git nodejs npm -y
```

Keep the project inside the WSL filesystem, for example:

```text
~/Grad_Project
```

Do **not** run it from `/mnt/c/...` if possible.

---

## 3. Create the Python Environments

This project uses **two Python environments**:

- `venv` → ResNet / MIL / Macenko / MIL+KAN
- `newvenv` → Vision Mamba / Vision Mamba + KAN

### Create `venv`
```bash
cd ~/Grad_Project
python3 -m venv venv
source venv/bin/activate
pip install -r backend_venv/requirments_venv.txt
deactivate
```

### Create `newvenv`
```bash
cd ~/Grad_Project
python3 -m venv newvenv
source newvenv/bin/activate
pip install -r backend_newvenv/requirments_newvenv.txt
deactivate
```

> Note: the repository uses the filenames:
>
> - `requirments_venv.txt`
> - `requirments_newvenv.txt`

---

## 4. Dataset Setup

Download the dataset images from:

```text
https://bmirds.github.io/KidneyCancer/
```

Use the project’s included metadata file:

```text
data/wsi_metadata.csv
```

Do **not** use the website `MetaData.csv`.

Create this folder:

```text
raw_dataset/
```

Place the downloaded WSI files inside it.

Example:

```text
Grad_Project/
  raw_dataset/
    DHMC_0001.png
    DHMC_0002.png
```

The filenames must match the `wsi_id` values in:

```text
data/wsi_metadata.csv
```

---

## 5. Prepare the Dataset

Run these steps from the project root:

```bash
cd ~/Grad_Project
python3 data_prep/slice_and_split_wsis.py
python3 data_prep/create_patch_csv.py
python3 Tools/audit_patch_whiteness.py
python3 data_prep/check_counts.py
python3 test_loader.py
```

This will create:

- patch folders inside `data/train`, `data/validate`, `data/test`
- `data/patch_metadata.csv`
- `patch_whiteness_audit.csv`

---

## 6. Two Ways to Use This Project

## Option A — Use the existing trained `.pth` files

If the repository already includes trained model weights, you can skip model training and go directly to:

- **Section 8: Run the Application**

The `.pth` files are expected inside each model folder, for example:

```text
Models/<ModelName>/results/training/
```

---

## Option B — Train the models yourself

If you train the models again, the new training run will save a new best model and **replace the old `.pth` file** inside that model’s training results folder.

Example output location:

```text
Models/<ModelName>/results/training/
```

### Run models in `venv`
```bash
source ~/Grad_Project/venv/bin/activate
cd ~/Grad_Project

python3 Models/ResNet18/Train_ResNet18.py
python3 Models/ResNet18/Test_ResNet18_MultiSample.py
python3 Models/ResNet18/ResNet18_visuals.py

python3 Models/ResNet18_MIL/Train_ResNet18_MIL.py
python3 Models/ResNet18_MIL/Test_ResNet18_MIL_MultiSample.py
python3 Models/ResNet18_MIL/ResNet18_MIL_visuals.py

python3 Models/ResNet18_MIL_Macenko/Train_ResNet18_MIL_Macenko.py
python3 Models/ResNet18_MIL_Macenko/Test_ResNet18_MIL_Macenko_MultiSample.py
python3 Models/ResNet18_MIL_Macenko/ResNet18_MIL_Macenko_visuals.py

python3 Models/ResNet18_MIL_KAN/Train_ResNet18_MIL_KAN.py
python3 Models/ResNet18_MIL_KAN/Test_ResNet18_MIL_KAN_MultiSample.py
python3 Models/ResNet18_MIL_KAN/ResNet18_MIL_KAN_visuals.py

deactivate
```

### Run models in `newvenv`
```bash
source ~/Grad_Project/newvenv/bin/activate
cd ~/Grad_Project

python3 Models/ResNet18_VisionMamba/Train_ResNet18_VisionMamba.py
python3 Models/ResNet18_VisionMamba/Test_ResNet18_VisionMamba_MultiSample.py
python3 Models/ResNet18_VisionMamba/ResNet18_VisionMamba_visuals.py

python3 Models/ResNet18_VisionMamba_KAN/Train_ResNet18_VisionMamba_KAN.py
python3 Models/ResNet18_VisionMamba_KAN/Test_ResNet18_VisionMamba_KAN_MultiSample.py
python3 Models/ResNet18_VisionMamba_KAN/ResNet18_VisionMamba_KAN_visuals.py

deactivate
```

---

## 7. Model Output Structure

Each model writes results inside:

```text
Models/<ModelName>/results/
```

Typical structure:

```text
training/
fixed_test/
multisample/
visuals/
```

---

## 8. Install Frontend Dependencies

```bash
cd ~/Grad_Project/frontend
npm install
```

---

## 9. Run the Application

Start **3 terminals**.

### Terminal 1 — `venv` backend
```bash
source ~/Grad_Project/venv/bin/activate
cd ~/Grad_Project/backend_venv
uvicorn app:app --host 0.0.0.0 --port 8001
```

### Terminal 2 — `newvenv` backend
```bash
source ~/Grad_Project/newvenv/bin/activate
cd ~/Grad_Project/backend_newvenv
uvicorn app:app --host 0.0.0.0 --port 8002
```

### Terminal 3 — frontend
```bash
cd ~/Grad_Project/frontend
npm run dev
```

Open the local URL shown by Vite, usually:

```text
http://localhost:5173/
```

---

## 10. How to Use the Web App

- Choose a model
- Choose the known correct label
- Upload patches from **one biopsy / one WSI only**
- Upload between **70 and 500 patches**

The backend will:

- analyze patch white/tissue content
- remove very empty patches
- sort by tissue percentage
- use the **best 70 patches**
- return the tumor prediction

MIL-based models can also return patch importance.

---

## 11. Optional: Compare All Models

After all models have been run, you can create a comparison table:

```bash
source ~/Grad_Project/venv/bin/activate
cd ~/Grad_Project
python3 compare_all_models.py
```

This creates comparison outputs inside:

```text
model_comparison/
```

---

## 12. Troubleshooting

### `Could not import module "app"`
Run uvicorn from the correct backend folder:

```bash
cd ~/Grad_Project/backend_venv
```

or

```bash
cd ~/Grad_Project/backend_newvenv
```

### Model file not found
Make sure the `.pth` file exists inside the correct model folder.

### Frontend opens but prediction fails
Make sure:

- backend 1 is running on `8001`
- backend 2 is running on `8002`
- frontend is running with `npm run dev`

### Not enough usable patches after filtering
Upload more patches from the same biopsy/WSI.

### VS Code is not using Ubuntu
Make sure the bottom-left corner says:

```text
WSL: Ubuntu
```
