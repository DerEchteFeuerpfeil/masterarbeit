# Find all SVG files in the current directory and subdirectories
find . -type f -name "*.svg" -print0 | while IFS= read -r -d $'\0' file; do
    # Convert SVG to PDF using Inkscape
    inkscape "$file" --export-type=pdf --export-filename="${file%.svg}.pdf"
    
    # Check if the PDF was created successfully
    if [ -f "${file%.svg}.pdf" ]; then
        # Delete the original SVG file
        rm "$file"
    else
        echo "Failed to convert $file"
    fi
done