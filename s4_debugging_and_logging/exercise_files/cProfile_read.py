import pstats
p = pstats.Stats('cProfile_out')
p.strip_dirs().sort_stats("tottime").print_stats()
