import torch
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class WTAClassifierConfiguration:
    noise_level: float = 0.5
    lr: float = 0.05
    classifier_lr: float = 0.01
    initial_bias: float = 1.0
    feedback_cf: float = 0.2
    constant_feedback: float = 0.0
    do_input_normalization: bool = True
    do_weights_normalization: bool = True
    temperature: float = 0.0
    temperature_decay: float = 0.9999

    def __post_init__(self):
        assert 0.0 <= self.noise_level <= 1.0
        assert 0.0 <= self.lr <= 1.0
        assert 0.0 <= self.classifier_lr <= 1.0
        assert 0.0 <= self.initial_bias
        assert 0.0 >= self.constant_feedback
        assert 0.0 <= self.temperature <= 1.0
        assert 0.0 <= self.temperature <= 1.0


class WTAClassifier(object):
    def __init__(
        self, n_inputs, n_clusters, n_classes,
        configuration: WTAClassifierConfiguration
    ):
        self._n_inputs = n_inputs
        self._n_clusters = n_clusters
        self._n_classes = n_classes
        self._configuration = configuration
        self._temperature = self._configuration.temperature

        self._clustering_weights = torch.ones([n_clusters, n_inputs], dtype=torch.float32)
        self._clustering_weights[:, :] -= 2 * torch.rand(
            [n_clusters, n_inputs], dtype=torch.float32
        ) * self._configuration.noise_level
        if self._configuration.do_weights_normalization:
            self._clustering_weights /= self._clustering_weights.norm(dim=-1, keepdim=True)
        self._classifier_weights = torch.ones([n_classes, n_clusters], dtype=torch.float32)
        self._classifier_weights[:, :] -= torch.rand(
            [n_classes, n_clusters], dtype=torch.float32
        )
        if self._configuration.constant_feedback < 0.0:
            self._inverse_classifier_weights = torch.full([n_clusters, n_classes], self._configuration.constant_feedback)
            clusters_per_class = n_clusters // n_classes
            cluster_idx = 0
            for c in range(n_classes):
                for _ in range(clusters_per_class):
                    mapping[cluster_idx, c] = 0.0
                    cluster_idx += 1
        else:
            self._inverse_classifier_weights = torch.ones([n_clusters, n_classes], dtype=torch.float32)

        self._clustering_bias = torch.ones([n_clusters], dtype=torch.float32) * self._configuration.initial_bias
        self._winners_stats = torch.zeros([n_clusters], dtype=torch.int32)

    def n_inputs(self):
        return self._n_inputs

    def n_clusters(self):
        return self._n_clusters

    def n_classes(self):
        return self._n_classes

    def get_device(self):
        return self._clustering_weights.device

    def to_device(self, device):
        self._clustering_weights = self._clustering_weights.to(device=device)
        self._classifier_weights = self._classifier_weights.to(device=device)
        self._inverse_classifier_weights = self._inverse_classifier_weights.to(device=device)
        self._clustering_bias = self._clustering_bias.to(device=device)
        self._winners_stats = self._winners_stats.to(device=device)

    def train(self, data, gt):
        with (torch.no_grad()):
            xb = data.reshape([data.shape[0], self._n_inputs])
            gtb = gt.reshape([gt.shape[0], self._n_classes])
            if self._configuration.do_input_normalization:
                xb = xb / (xb.norm(dim=-1, keepdim=True) + 1e-08)

            u = torch.matmul(xb, self._clustering_weights.permute(1, 0))
            biased_u = u + self._clustering_bias

            if self._configuration.feedback_cf > 0.0:
                inverse_prediction = gtb @ self._inverse_classifier_weights.T
                winners = torch.argmax(biased_u + self._configuration.feedback_cf * inverse_prediction, dim=-1, keepdim=True)
            else:
                winners = torch.argmax(biased_u, dim=-1, keepdim=True)

            y_k = torch.zeros([data.shape[0], self._n_clusters], device=xb.device)
            y_k.scatter_(1, winners, 1.0)

            if self._configuration.constant_feedback == 0.0:
                inverse_d = gtb.unsqueeze(1) - self._inverse_classifier_weights
                inverse_delta_w = (inverse_d * y_k.unsqueeze(2)).sum(dim=0)
                self._inverse_classifier_weights += inverse_delta_w * self._configuration.lr

            prediction = y_k @ self._classifier_weights.T
            grad_classifier = prediction - gtb
            classifier_delta_w = -(grad_classifier.T @ y_k)
            self._classifier_weights += classifier_delta_w * self._configuration.classifier_lr

            self._winners_stats += y_k.sum(dim=0).to(dtype=torch.int32)
            if self._temperature > 0.0:
                quasi_grad = -torch.softmax(biased_u / self._anti_hebb_coeff, dim=-1)
                quasi_grad.scatter_(1, winners, 1.0)
                self._temperature *= self._configuration.temperature_decay
            else:
                quasi_grad = y_k

            d = xb.unsqueeze(1) - self._clustering_weights
            clustering_delta_w = (d * quasi_grad.unsqueeze(2)).sum(dim=0)
            delta_b = (quasi_grad * (0.0 - self._clustering_bias)).sum(dim=0)

            self._clustering_weights += self._configuration.lr * clustering_delta_w
            if self._configuration.do_weights_normalization:
                self._clustering_weights /= self._clustering_weights.norm(dim=-1, keepdim=True)
            self._clustering_bias += delta_b * self._configuration.lr
            return biased_u, y_k

    def classify(self, data, use_bias=True):
        with torch.no_grad():
            xb = data.reshape([data.shape[0], self._n_inputs])
            if self._configuration.do_input_normalization:
                xb = xb / (xb.norm(dim=-1, keepdim=True) + 1e-08)

            u = torch.matmul(xb, self._clustering_weights.permute(1, 0))
            if use_bias:
                biased_u = u + self._clustering_bias
            else:
                biased_u = u

            winners = torch.argmax(biased_u, dim=-1, keepdim=True)
            y_k = torch.zeros([data.shape[0], self._n_clusters], device=xb.device)
            y_k.scatter_(1, winners, 1.0)
            prediction = y_k @ self._classifier_weights.T
            return y_k, prediction

    def get_last_mean_delta_w(self):
        return self._last_mean_delta_w

    def get_winners_stats(self):
        return self._winners_stats

    def get_cluster_weights(self):
        with torch.no_grad():
            return self._clustering_weights.clone()


def measure_classifier_accuracy(
        classifier, data_loader, max_samples=None, discard_bias=False
):
    hits = 0.0
    total = 0.0
    for batch_idx, (source_data, gt) in enumerate(data_loader):
        gt_one_hot = torch.nn.functional.one_hot(gt, num_classes=10).to(dtype=torch.float32).cpu().detach()
        _, prediction = classifier.classify(source_data, not discard_bias)
        prediction_one_hot = torch.nn.functional.one_hot(prediction.max(dim=-1)[1], num_classes=10).to(
            dtype=torch.float32).cpu().detach()
        hits += float(torch.all(gt_one_hot == prediction_one_hot, dim=1).sum().item())
        total += source_data.shape[0]
        if max_samples is not None and total >= max_samples:
            break

    return hits / total
