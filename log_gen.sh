#!/bin/bash
output_file="log_agg.txt"
unique_names_file="unique_names.txt"
unique_moves_file="unique_moves.txt"

format_log_entry() {
  sed -n -e '/Slot:/,/^$/p' "$1" | sed 's/^/  /'
}
find . -type d -name "session_*" -print0 | while IFS= read -r -d '' session_dir; do
  find "$session_dir" -type f -name "*.txt" -print0 | while IFS= read -r -d '' log_file; do
    format_log_entry "$log_file" | grep -o 'Name:.*' | sed 's/Name: //' >> "$unique_names_file"
    format_log_entry "$log_file" | grep -o 'Moves:.*' | sed 's/Moves: //' | tr ',' '\n' | sed '/^$/d' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//' >> "$unique_moves_file"
  done
done
sort -u "$unique_names_file" > "$unique_names_file.tmp"
sort -u "$unique_moves_file" > "$unique_moves_file.tmp"
mv "$unique_names_file.tmp" "$unique_names_file"
mv "$unique_moves_file.tmp" "$unique_moves_file"
{
  echo "============= Unique Pokemon ============="
  cat "$unique_names_file"
  echo
  echo "============== Unique Moves =============="
  echo 
  cat "$unique_moves_file"
  echo
  echo "================== Log Entries =================="

  find . -type d -name "session_*" -print0 | while IFS= read -r -d '' session_dir; do
    echo "==============${session_dir}=============="
    find "$session_dir" -type f -name "*.txt" -print0 | while IFS= read -r -d '' log_file; do
      format_log_entry "$log_file"
    done
  done
} > "$output_file"
rm "$unique_names_file" "$unique_moves_file"

echo "Done..."