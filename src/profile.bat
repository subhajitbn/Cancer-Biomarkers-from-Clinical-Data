@echo off
REM Profile the Python script and save profiling data to a temporary file
python -m cProfile -o program.prof %1

REM Filter the profiling data, keeping only the top 20 cumulative time, and save to a new file
python -c "import pstats; p = pstats.Stats('program.prof'); p.sort_stats('cumtime'); p.print_stats(20)"