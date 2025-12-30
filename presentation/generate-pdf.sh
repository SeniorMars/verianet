#!/bin/bash

# Generate PDF from print-friendly HTML
# Usage: ./generate-pdf.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PRINT_HTML="file://$SCRIPT_DIR/print.html"
OUTPUT_PDF="$SCRIPT_DIR/presentation.pdf"

echo "Generating PDF from $PRINT_HTML..."

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    --headless \
    --disable-gpu \
    --print-to-pdf="$OUTPUT_PDF" \
    --no-pdf-header-footer \
    --print-to-pdf-no-header \
    "$PRINT_HTML" 2>&1

if [ -f "$OUTPUT_PDF" ]; then
    SIZE=$(du -h "$OUTPUT_PDF" | cut -f1)
    echo "PDF generated successfully: $OUTPUT_PDF ($SIZE)"
else
    echo "Error: PDF generation failed"
    exit 1
fi
