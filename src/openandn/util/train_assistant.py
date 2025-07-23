import torch


class WTAClusteriserMeasureAccuracyByVotingAction(object):
    def __init__(
        self, n_gt_classes,
        test_dataset, train_dataset,
        use_bias=False, batch_size=512
    ):
        self._n_gt_classes = n_gt_classes
        self._test_dataset = test_dataset
        self._train_dataset = train_dataset
        self._batch_size = batch_size
        self._use_bias = use_bias

    def __call__(self, clusteriser):
        data_loader = torch.utils.data.DataLoader(
            self._train_dataset, batch_size=self._batch_size, shuffle=True
        )

        clusters_stat = torch.zeros(
            [clusteriser.n_clusters(), self._n_gt_classes], dtype=torch.int32, device=clusteriser.get_device()
        )
        for source_data, gt in data_loader:
            winners = clusteriser.clusterize(source_data, self._use_bias)[0].argmax(dim=-1)
            if len(winners.shape) == 2:
                winners = winners.squeeze(-1)
            flat_indices = winners.long() * self._n_gt_classes + gt.long()
            clusters_stat.view(-1).scatter_add_(
                0, flat_indices, torch.ones_like(flat_indices, dtype=torch.int32)
            )

        predicted_labels = clusters_stat.argmax(dim=1)

        def compute_accuracy(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
            total_correct, total_samples = 0, 0
            for source_data, gt in loader:
                winners = clusteriser.clusterize(source_data, self._use_bias)[0].argmax(dim=-1)
                if len(winners.shape) == 2:
                    winners = winners.squeeze(-1)
                preds = predicted_labels[winners]
                total_correct += (preds == gt).sum().item()
                total_samples += gt.size(0)
            return total_correct / total_samples

        train_accuracy = compute_accuracy(self._train_dataset)
        test_accuracy = compute_accuracy(self._test_dataset)
        return train_accuracy, test_accuracy, predicted_labels
