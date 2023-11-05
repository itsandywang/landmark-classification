def update_best_epoch(epoch, min_validation_loss, best_epoch_stats, best_epoch, stats):
    if stats[-1][1] < min_validation_loss:
        min_validation_loss = stats[-1][1]
        best_epoch_stats = stats[-1]
        best_epoch = epoch + 1
        print(f"Best epoch is now: {best_epoch}")
    return best_epoch, best_epoch_stats, min_validation_loss

def print_best_epoch_stats(best_epoch, best_epoch_stats, stats):
    print(f"Epoch with lowest Validation Loss:")
    splits = ["Validation", "Train", "Test"]
    metrics = ["Accuracy", "Loss", "AUROC"]
    print("Epoch {}".format(best_epoch))
    for j, split in enumerate(splits):
        for i, metric in enumerate(metrics):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            print(f"\t{split} {metric}:{round(best_epoch_stats[idx],4)}")