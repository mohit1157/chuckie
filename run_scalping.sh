#!/bin/bash
echo "========================================"
echo "  High-Probability Scalping Bot"
echo "  Targeting 80%+ Win Rate"
echo "========================================"
echo

# Check if config argument provided
if [ -z "$1" ]; then
    echo "Using default config: configs/scalping.yaml"
    CONFIG="configs/scalping.yaml"
else
    CONFIG="$1"
fi

echo "Config: $CONFIG"
echo
echo "Starting bot..."
echo "Press Ctrl+C to stop"
echo

python -m bot.scalping_bot --config "$CONFIG"
