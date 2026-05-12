# run all vtk files in tetra_files directory
mkdir -p training_data
for file in tetra_files/*_2.vtk; do
  echo "Processing $file"
  MeshabilityEvaluator -i "$file" -o "training_data/$(basename "$file" .vtk).json"
done