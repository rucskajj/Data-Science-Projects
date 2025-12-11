
start_year=2015
end_year=2024
test_year=2024

#echo "python ratioBasedxG.py -sy $start_year -ey $end_year -ty $test_year"

for ((test_year = start_year; test_year <= end_year; test_year++)); do
    python ratioBasedxG.py -sy $start_year -ey $end_year -ty $test_year
done