#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "Syncing the locked Python 3.12 environment..."
uv sync --locked --python 3.12
mkdir -p data

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to verify your app token. Install it and run setup again." >&2
    exit 1
fi

verify_token() {
    local status
    echo "Verifying app token..."
    if ! status=$(curl --silent --show-error --connect-timeout 10 --max-time 30 \
        --output /dev/null --write-out '%{http_code}' \
        --header "X-App-Token: $1" \
        'https://data.cityofchicago.org/resource/85ca-t3if.json?$limit=1'); then
        echo "Could not reach the Chicago Data Portal. Check your connection and rerun setup." >&2
        exit 1
    fi
    case "$status" in
        200) echo "App token verified." ;;
        400|401|403)
            echo "The portal rejected the token (HTTP $status). Please enter another token." >&2
            return 1
            ;;
        *)
            echo "Token verification failed (HTTP $status). Try running setup again later." >&2
            exit 1
            ;;
    esac
}

token=$(
    unset CHICAGO_DATA_APP_TOKEN
    if [[ -f data/.env ]]; then
        source data/.env
    fi
    printf '%s' "${CHICAGO_DATA_APP_TOKEN:-}"
)
if [[ -n "$token" ]] && verify_token "$token"; then
    echo "Using the existing app token in data/.env."
    unset token
else
    echo
    echo "Sign up for the Chicago Data Portal: https://data.cityofchicago.org/signup"
    echo "Then create an app token in your profile: https://data.cityofchicago.org/profile/edit/developer_settings"
    while true; do
        read -r -s -p "Paste your Chicago portal app token (input hidden): " token
        printf '\n'
        if [[ -n "${token//[[:space:]]/}" ]]; then
            if verify_token "$token"; then
                break
            fi
        else
            echo "Please enter a non-empty app token."
        fi
    done

    touch data/.env
    chmod 600 data/.env
    printf '\nCHICAGO_DATA_APP_TOKEN=%q\n' "$token" >> data/.env
    unset token
    echo "Saved app token to data/.env."
fi

echo
echo "Downloading the graph data will take a while and requires ~2 GB of disk space."
while true; do
    read -r -p "Download now? [y/N] " download
    case "$download" in
        y|Y|yes|YES|Yes)
            if ! command -v curl >/dev/null 2>&1; then
                echo "curl is required to download the data. Install it and run bash scripts/fetch_data.sh." >&2
                exit 1
            fi
            PYTHON_PATH="$PROJECT_DIR/.venv/bin/python" bash "$PROJECT_DIR/scripts/fetch_data.sh"
            break
            ;;
        n|N|no|NO|No|"") break ;;
        *) echo "Please answer yes or no." ;;
    esac
done

echo "Setup complete. Activate the environment with: source .venv/bin/activate"
