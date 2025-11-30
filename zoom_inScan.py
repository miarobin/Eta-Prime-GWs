import pandas as pd
import numpy as np
from pathlib import Path

#Dont forget to set up the path file!

def process_one_file(
    csv_path,
    out_folder,
    alpha_col="Alpha",
    beta_col="Beta",
    param_cols=("m2Sig", "m2Eta", "m2X", "I"),
):
    csv_path = Path(csv_path)
    out_folder = Path(out_folder)

    print(f"Processing {csv_path.name}...")

    # Read CSV, skipping malformed rows
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    # 1) Drop any rows with NaNs
    df_full = df.dropna(how="any")

    # 2) Drop rows with NO transition: Tc == 0 AND Tn == 0
    if {"Tc", "Tn"}.issubset(df_full.columns):
        df_full = df_full[~((df_full["Tc"] == 0) & (df_full["Tn"] == 0))]

    if df_full.empty:
        print("  -> No valid (transition) rows, skipping.")
        return

    # Make sure output folder exists
    out_folder.mkdir(parents=True, exist_ok=True)

    #  FILE 1: all datapoints with full info & transition 
    datapoints_name = out_folder / f"{csv_path.stem}_datapoints.csv"
    df_full.to_csv(datapoints_name, index=False)
    print(f"  -> Saved {len(df_full)} transition rows to {datapoints_name.name}")

    # Check columns exist for alpha/beta
    if alpha_col not in df_full.columns or beta_col not in df_full.columns:
        print(
            f"  -> WARNING: '{alpha_col}' or '{beta_col}' not in columns, "
            "skipping best 20% selection."
        )
        return

    # 3) Compute cuts on Alpha & Beta using the transition rows
    alpha_cut = df_full[alpha_col].quantile(0.6)  # top 40% Alpha
    beta_cut = df_full[beta_col].quantile(0.4)    # bottom 40% Beta

    # INTERSECTION: high Alpha & low Beta 
    mask = (df_full[alpha_col] >= alpha_cut) & (df_full[beta_col] <= beta_cut)
    df_best = df_full[mask]

    if df_best.empty:
        print("  -> No rows pass alpha/beta cuts.")
        return

    # FILE 2: best % intersection 
    betterscan_name = out_folder / f"betterscan_{csv_path.stem}.csv"
    df_best.to_csv(betterscan_name, index=False)
    print(f"  -> Saved {len(df_best)} best-range rows to {betterscan_name.name}")

    #Core ranges (10–90% quantiles) + top points to a file
    summary_path = out_folder / f"ranges_{csv_path.stem}.txt"
    with open(summary_path, "w") as f:
        f.write(f"File: {csv_path.name}\n")
        f.write("Core parameter ranges from best points (10–90% quantiles):\n")

        for col in param_cols:
            if col not in df_best.columns:
                f.write(f"  (column '{col}' not found, skipping)\n")
                continue

            q10 = df_best[col].quantile(0.20)
            q90 = df_best[col].quantile(0.80)
            med = df_best[col].median()

            if col.startswith("m2"):
                #scan as np.linspace(a, b, num)*1000**2
                f.write(
                    f"  {col}: {q10/1e6:.3g} – {q90/1e6:.3g} "
                    f"(median ~ {med/1e6:.3g}, for np.linspace(...)*1000**2)\n"
                )
            elif col == "fPI":
                f.write(
                    f"  {col}: {q10:.3g} – {q90:.3g} GeV "
                    f"(median ~ {med:.3g} GeV)\n"
                )
            else:
                f.write(
                    f"  {col}: {q10:.3g} – {q90:.3g} "
                    f"(median ~ {med:.3g})\n"
                )

        # Define a "score": high Alpha, low Beta
        a = df_best[alpha_col]
        b = df_best[beta_col]
        a_norm = (a - a.min()) / (a.max() - a.min() + 1e-12)
        b_norm = (b.max() - b) / (b.max() - b.min() + 1e-12)

        df_best_scored = df_best.copy()
        df_best_scored["score"] = a_norm + b_norm
        df_top = df_best_scored.sort_values("score", ascending=False).head(5)

        f.write("\nTop 5 best individual points:\n")
        for i, (_, row) in enumerate(df_top.iterrows(), start=1):
            param_str = ", ".join(
                f"{col}={row[col]:.3g}"
                for col in param_cols
                if col in df_best_scored.columns
            )
            f.write(
                f"  #{i}: {param_str} | "
                f"{alpha_col}={row[alpha_col]:.3g}, "
                f"{beta_col}={row[beta_col]:.3g}\n"
            )

    print(f"  -> Saved summary to {summary_path}\n")


def process_folder(base_folder, alpha_col="Alpha", beta_col="Beta"):
    base_folder = Path(base_folder)
    out_folder = base_folder / "bestranges"

    print("file")
    for csv_path in base_folder.glob("*.csv"):
        process_one_file(
            csv_path,
            out_folder,
            alpha_col=alpha_col,
            beta_col=beta_col,
            param_cols=("m2Sig", "m2Eta", "m2X", "fPI"),
        )


if __name__ == "__main__":
    base = Path("/GW-eta'-mass/F3/N3"). ##Set up the path
    process_folder(base, alpha_col="Alpha", beta_col="Beta")
