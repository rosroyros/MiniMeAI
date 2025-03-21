#!/bin/bash

# Simple script to trigger vector DB optimization
echo "Triggering full optimization of vector collection..."
curl -X POST "http://localhost:8000/api/v1/collections/emails/optimize" \
  -H "Content-Type: application/json" \
  -d '{"target_dim": 128}'

echo -e "\n\nChecking optimization status..."
curl "http://localhost:8000/api/v1/collections/emails/status"
echo -e "\n" 