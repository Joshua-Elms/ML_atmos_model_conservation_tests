import requests
from pathlib import Path

download_dir = Path(
    "/N/slate/jmelms/projects/ML_atmos_model_conservation_tests/A.mass_conservation/run_ACE2/model_data"
)
repo_url = "https://huggingface.co/allenai/ACE2-ERA5/resolve/main/"
download_year = (
    2020  # options: 1940, 1950, 1979, 2001, 2020 (ics only available for these years)
)
download_items = {
    "ace2_era5_ckpt.tar": repo_url + "ace2_era5_ckpt.tar",
    "inference_config.yaml": repo_url + "inference_config.yaml",
    f"initial_conditions/ic_{download_year}.nc": repo_url + f"initial_conditions/ic_{download_year}.nc",
    f"forcings/forcing_{download_year}.nc": repo_url + f"forcing_data/forcing_{download_year}.nc",
}
download_dir.mkdir(parents=False, exist_ok=True)

for item_name, item_url in download_items.items():
    print(f"Downloading {item_name}...")
    response = requests.get(item_url)
    response.raise_for_status()  # Ensure we notice bad responses
    with open(download_dir / item_name, "wb") as f:
        f.write(response.content)
    print(f"Saved to {download_dir / item_name}")
