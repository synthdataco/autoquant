#!/usr/bin/env bash
# Launches the campaign_3_gemini-3-8-flash miner on Bittensor Subnet 50.
# Configurable via env vars. Defaults target mainnet finney.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NETWORK="${NETWORK:-finney}"
NETUID="${NETUID:-50}"
COLDKEY_NAME="${COLDKEY_NAME:-miner}"
HOTKEY_NAME="${HOTKEY_NAME:-default}"
PORT="${PORT:-8091}"
VALIDATOR_MIN_STAKE="${VALIDATOR_MIN_STAKE:-65000}"

exec uv run python3.12 "${SCRIPT_DIR}/miner.py" \
    --subtensor.network "$NETWORK" \
    --netuid "$NETUID" \
    --logging.debug \
    --wallet.name "$COLDKEY_NAME" \
    --wallet.hotkey "$HOTKEY_NAME" \
    --axon.port "$PORT" \
    --blacklist.force_validator_permit true \
    --blacklist.validator_min_stake "$VALIDATOR_MIN_STAKE"
