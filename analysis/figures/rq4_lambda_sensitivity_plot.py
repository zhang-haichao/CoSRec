from pathlib import Path

from experiments.rq4.lambda_sensitivity import plot_results_three_panel


def main() -> None:
    plot_results_three_panel(
        Path("results/paper/rq4/rq4_lambda_sensitivity.csv"),
        Path("results/paper/rq4/rq4_lambda_sensitivity_1x3.pdf"),
    )
    plot_results_three_panel(
        Path("results/paper/rq4/rq4_lambda_sensitivity.csv"),
        Path("results/paper/rq4/rq4_lambda_sensitivity_1x3.png"),
    )


if __name__ == "__main__":
    main()
