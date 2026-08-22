#!/bin/bash
# Double-click this to run the demo properly (so Transfer works).
cd "$(dirname "$0")"
echo "Starting the demo server..."
echo "Your browser will open. Keep this window open while using the demo."
echo "When you're done, close this window to stop the server."
( sleep 1; open "http://localhost:8000" ) &
python3 -m http.server 8000
