import pstats

p = pstats.Stats("output.txt")
p.sort_stats('cumulative')
p.strip_dirs().sort_stats(-1).print_stats()
