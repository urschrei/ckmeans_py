import cProfile
import profile
import pstats

import numpy as np

print("Calibrating")
# calibrate
pr = profile.Profile()
calibration = np.mean([pr.calibrate(100000) for x in range(5)])
# add the bias
profile.Profile.bias = calibration

# protect the entry point
if __name__ == "__main__":
    print("Test: 110k random uniform samples, 1.0 >= samples <= 3.0,  5 clusters")
    with open("tests/bench_rust.py", "rb") as f1:
        rust = f1.read()

    with open("tests/bench_cpp.py", "rb") as f2:
        cpp = f2.read()
    print("Running Rust")
    cProfile.run(rust, "tests/output_stats_rust")
    print("Running cpp")
    cProfile.run(cpp, "tests/output_stats_cpp")
    rust_result = pstats.Stats("tests/output_stats_rust")
    cpp_result = pstats.Stats("tests/output_stats_cpp")

    rust_result_sorted = rust_result.sort_stats("tottime").print_stats(5)
    cpp_result_sorted = cpp_result.sort_stats("tottime").print_stats(5)
