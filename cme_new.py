import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def parse_numeric(value):
    """Parse numeric fields that may contain '*', '-------', or scientific notation."""
    value = value.strip()

    # Handle '-------'
    if value.startswith("-") and not value.replace("-", "").replace(".", "").replace("e", "").isdigit():
        return None

    # Remove trailing asterisk notation
    value = value.replace("*", "")

    try:
        return float(value)
    except ValueError:
        return None


def fetch_cme_data(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    month = date.month

    fname = f"univ{year}_{month:02d}.txt"
    base_url = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/text_ver/"
    url = base_url + fname

    print(f"Fetching: {url}")
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError(f"Could not download file: {url}")

    text = r.text
    target_date = date.strftime("%Y/%m/%d")

    rows = []
    for line in text.splitlines():
        if line.startswith(target_date):
            rows.append(line.strip())

    if not rows:
        print("No CME entries found for this date.")
        return None

    parsed = []

    for row in rows:
        parts = row.split()

        date_val = parts[0]
        time_val = parts[1]

        # CPA and Width handling (Halo)
        cpa_raw = parts[2]
        if cpa_raw == "Halo":
            cpa = None
            width = 360.0
            offset = 1
        else:
            cpa = parse_numeric(cpa_raw)
            width = parse_numeric(parts[3])
            offset = 0

        lin_speed = parse_numeric(parts[3 + offset])
        init_speed = parse_numeric(parts[4 + offset])
        final_speed = parse_numeric(parts[5 + offset])
        speed_20R = parse_numeric(parts[6 + offset])

        accel = parse_numeric(parts[7 + offset])
        mass = parse_numeric(parts[8 + offset])
        kinetic = parse_numeric(parts[9 + offset])

        mpa_raw = parts[10 + offset]
        mpa = None if mpa_raw == "Halo" else parse_numeric(mpa_raw)

        remark = " ".join(parts[11 + offset:]) if len(parts) > 11 + offset else ""

        parsed.append({
            "date": date_val,
            "time": time_val,
            "CPA": cpa,
            "Width": width,
            "Linear_Speed": lin_speed,
            "Initial_Speed": init_speed,
            "Final_Speed": final_speed,
            "Speed_20R": speed_20R,
            "Acceleration": accel,
            "Mass": mass,
            "Kinetic_Energy": kinetic,
            "MPA": mpa,
            "Remarks": remark
        })

    df = pd.DataFrame(parsed)

    # Convert time to full datetime
    df["Datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

    return df


def plot_speed_timeseries(df, date_str):
    plt.figure(figsize=(10, 4))

    plt.scatter(df["Datetime"], df["Linear_Speed"], marker="o", linewidth=2)

    plt.title(f"CME Linear Speeds on {date_str}")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Linear Speed (km/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    date = "2012-01-01"  # Example

    df = fetch_cme_data(date)

    if df is not None:
        print(df)
        plot_speed_timeseries(df, date)
