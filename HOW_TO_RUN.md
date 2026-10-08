# How to Run the Code

## 1) Prerequisites
- Python 3.9+ (recommended)
- `pip`

## 2) Install Dependencies
From the repository root:

```bash
pip install -r requirements.txt
```

## 3) Run the Demo
Default run (30 tasks, 2 UGVs, 3 UAVs):

```bash
python demo.py
```

Custom run (`n m_g m_a`):

```bash
python demo.py 60 3 4
```

## 4) Output Files
After a successful run, the script saves:
- `routes.png`
- `objective_space.png`
