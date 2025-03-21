#!/bin/bash

# Script to collect all MiniMeAI project files into a single timestamped text file
# Usage: ./collect_project_files.sh

output_file="minimai_project_files_$(date +"%Y%m%d_%H%M%S").txt"

echo "Collecting MiniMeAI project files into $output_file..."

(
  echo "====== MINIUME AI PROJECT FILES ======"
  echo "Generated: $(date)"
  echo ""
  
  # Main source files
  for file in \
    src/email/email_fetcher.py \
    src/processing/processor.py \
    src/processing/chroma_client.py \
    src/query/query_client.py \
    src/api/api_server.py \
    src/custom_vector_db.py \
    src/web/web_server.py
    src/web/static/* \
    src/template/* \
    docker-compose.yml \
    Dockerfile.email \
    Dockerfile.processor \
    Dockerfile.api \
    Dockerfile.vectordb \
    Dockerfile.web \
    .env
  do
    if [ -f "$file" ]; then
      echo "======================================="
      echo "FILE: $file"
      echo "======================================="
      cat "$file"
      echo ""
      echo ""
    elif [ -d "$(dirname "$file")" ]; then
      # For wildcards like config/*, check if directory exists
      for subfile in $file; do
        if [ -f "$subfile" ]; then
          echo "======================================="
          echo "FILE: $subfile"
          echo "======================================="
          cat "$subfile"
          echo ""
          echo ""
        fi
      done
    else
      echo "======================================="
      echo "FILE: $file (NOT FOUND)"
      echo "======================================="
      echo ""
    fi
  done
) > "$output_file"

echo "Collection complete. Output saved to $output_file"
