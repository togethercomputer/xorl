import os
from functools import lru_cache


@lru_cache
def is_env_option_enabled(opt: str) -> bool:
    return int(os.getenv(opt, "0"))


def is_assertion_enabled():
    return is_env_option_enabled("BPEX_DEBUG")


def is_untuned_warning_suppressed():
    return is_env_option_enabled("BPEX_NO_WARN_ON_UNTUNED_CASE") or testing_is_ci_env()


def debugging_fake_benchmark_result():
    return is_env_option_enabled("BPEX_DEBUGGING_FAKE_BENCHMARK_RESULT")


def debugging_is_verbose():
    return is_env_option_enabled("BPEX_DEBUGGING_VERBOSE")


def testing_is_ci_env():
    return is_env_option_enabled("BPEX_TESTING_IS_CI_ENV")


def testing_no_noncontiguous_tensors():
    return is_env_option_enabled("BPEX_TESTING_NO_NONCONTIGUOUS_TENSORS")


def benchmarking_minimal_run():
    return is_env_option_enabled("BPEX_BENCHMARKING_MINIMAL_RUN") or benchmarking_using_ncu()


def benchmarking_no_baseline():
    return is_env_option_enabled("BPEX_BENCHMARKING_NO_BASELINE") or benchmarking_using_ncu()


def benchmarking_using_ncu():
    return is_env_option_enabled("BPEX_BENCHMARKING_USE_NCU")


def benchmarking_write_report():
    return is_env_option_enabled("BPEX_BENCHMARKING_WRITE_REPORT")


def tuning_correctness_check_only():
    return is_env_option_enabled("BPEX_TUNING_CORRECTNESS_CHECK_ONLY")
