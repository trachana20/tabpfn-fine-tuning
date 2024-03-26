import matplotlib.pyplot as plt


def visualize_accuracy_accross_seeds(results_dict):
    for random_state, folds in results_dict.items():
        for fold_i, fold in folds.items():
            pre_eval = fold["pre_eval"]
            post_eval = fold["post_eval"]

            plt.plot(pre_eval["accuracy"], label=f"Pre-tuning {random_state} {fold_i}")
            plt.plot(
                post_eval["accuracy"], label=f"Post-tuning {random_state} {fold_i}"
            )

    plt.legend()
    plt.show()
