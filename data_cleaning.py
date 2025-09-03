# data_cleaning.py
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent
CDIR = ROOT / "raw_data"
OUTDIR = ROOT / "clean_data"
FF5 = CDIR / "F-F_Research_Data_5_Factors_2x3_daily.csv"
MOM = CDIR / "F-F_Momentum_Factor_daily.csv"
OUT = OUTDIR / "ff_factors_daily_clean.csv"
START = pd.Timestamp("2010-01-01")

def _header_line(path, targets):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            row = [c.strip() for c in line.strip().split(",")]
            if len(row) > 1 and row[1].replace(" ", "").lower() in targets:
                return i
    raise ValueError(f"Header não encontrado em {path.name}")

def load_ff5_daily(path: Path) -> pd.DataFrame:
    skip = _header_line(path, {"mkt-rf", "rm-rf"})
    df = pd.read_csv(path, skiprows=skip, header=0, encoding="utf-8")
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df = df[df["Date"].astype(str).str.fullmatch(r"\d{6,8}")].copy()
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.pad(8, fillchar="0"), format="%Y%m%d")
    df.rename(columns={"Mkt-RF": "MKT_RF", "MKT-RF": "MKT_RF", "Rm-Rf": "MKT_RF", "Rf": "RF"}, inplace=True)
    cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce") / 100.0
    df = df.set_index("Date").sort_index()[cols].dropna(how="any")
    return df.loc[df.index >= START]

def load_mom_daily(path: Path) -> pd.DataFrame:
    skip = _header_line(path, {"mom"})
    df = pd.read_csv(path, skiprows=skip, header=0, encoding="utf-8")
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df = df[df["Date"].astype(str).str.fullmatch(r"\d{6,8}")].copy()
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.pad(8, fillchar="0"), format="%Y%m%d")
    mom_col = [c for c in df.columns if str(c).strip().lower().startswith("mom")][0]
    df.rename(columns={mom_col: "MOM"}, inplace=True)
    df["MOM"] = pd.to_numeric(df["MOM"], errors="coerce") / 100.0
    df = df.set_index("Date").sort_index()[["MOM"]].dropna()
    return df.loc[df.index >= START]

if __name__ == "__main__":
    ff5 = load_ff5_daily(FF5)
    mom = load_mom_daily(MOM)
    factors = ff5.join(mom, how="inner").round(5)
    factors.to_csv(OUT, float_format="%.5f")
    print(f"Salvo: {OUT} | linhas={len(factors)} | {factors.index.min()} → {factors.index.max()}")
