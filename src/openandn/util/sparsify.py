import torch
import math
from random import shuffle


class SingleChannelPixelSparsifier(object):
    def __init__(self, reference_points, treat_zero_as_zero=True):
        self._use_cache = False
        self._reference_points = torch.tensor(reference_points, dtype=torch.float32)
        self._treat_zero_as_zero = treat_zero_as_zero
        assert len(self._reference_points.shape) == 1 and self._reference_points.shape[0] > 0
        if not treat_zero_as_zero:
            self._reference_points = torch.cat((torch.tensor([0.0], dtype=torch.float32), self._reference_points))
        self._reference_points_map = {}

    def set_use_cache(self, use_cache):
        self._use_cache = use_cache

    def _repeat_reference_points(self, n_pixels, device):
        if self._use_cache:
            key = (n_pixels, device,)
            res = self._reference_points_map.get(key)
            if res is not None:
                return res
        else:
            key = None

        res = self._reference_points.to(device=device)
        res = res.repeat(n_pixels, 1)

        if self._use_cache:
            self._reference_points_map[key] = res
        return res

    def __call__(self, pixels):
        n_pixels = pixels.shape[0]
        assert len(pixels.shape) == 1
        res = self._repeat_reference_points(n_pixels, pixels.device)
        pixels = pixels.unsqueeze(-1)
        res = (res - pixels).abs()
        mn = res.min(dim=-1, keepdim=True)
        res = torch.zeros(res.shape, device=res.device, dtype=torch.float32).scatter_(-1, mn[1], mn[0].clip(0.0000001))
        if self._treat_zero_as_zero:
            mx = pixels.max(dim=-1, keepdim=True)
            res[res >= mx[0]] = 0.0
        return res.ceil().reshape([n_pixels * self._reference_points.shape[0]])

    def reverse(self, sparse_pixels):
        n = sparse_pixels.shape[0]
        assert len(sparse_pixels.shape) == 1
        assert (n % self._reference_points.shape[0]) == 0
        n_pixels = n // self._reference_points.shape[0]
        ref_points = self._repeat_reference_points(n_pixels, sparse_pixels.device)
        return (sparse_pixels.reshape([n_pixels, self._reference_points.shape[0]]) * ref_points).sum(dim=1)


class MultiChannelPixelSparsifier(object):
    def __init__(self, reference_vectors, treat_zero_as_zero=True):
        self._use_cache = False
        self._reference_vectors = reference_vectors
        self._treat_zero_as_zero = treat_zero_as_zero
        assert len(self.reference_vectors.shape) == 2 and self.reference_vectors.shape[0] > 0
        if not treat_zero_as_zero:
            self._reference_vectors = torch.cat((torch.zeros([reference_vectors.shape[1]], dtype=torch.float32), self._reference_vectors))
        self._reference_vectors_map = {}

    def set_use_cache(self, use_cache):
        self._use_cache = use_cache

    def _repeat_reference_vectors(self, n_pixels, device):
        if self._use_cache:
            key = (n_pixels, device,)
            res = self._reference_vectors_map.get(key)
            if res is not None:
                return res
        else:
            key = None

        res = self._reference_vectors.to(device=device)
        res = res.repeat(n_pixels, 1, 1)

        if self._use_cache:
            self._reference_vectors_map[key] = res
        return res

    def __call__(self, pixels):
        n_pixels, c = pixels.shape
        res = self._repeat_reference_vectors(n_pixels, pixels.device)
        pixels = pixels.unsqueeze(-2).repeat(1, res.shape[-2], 1)
        res = ((pixels - res) ** 2).sum(dim=-1)
        mn = res.min(dim=-1, keepdim=True)
        res = torch.zeros(res.shape, device=res.device, dtype=torch.float32).scatter_(-1, mn[1], mn[0].clip(0.0000001))
        if self._treat_zero_as_zero:
            mx = (pixels**2).sum(dim=-1).max(dim=-1, keepdim=True)
            res[res >= mx[0]] = 0.0
        return res.ceil().reshape([n_pixels * self._reference_vectors.shape[0]])

    def reverse(self, sparse_pixels):
        n = sparse_pixels.shape[0]
        assert len(sparse_pixels.shape) == 1
        assert (n % self._reference_vectors.shape[0]) == 0
        n_pixels = n // self._reference_vectors.shape[0]
        ref_vectors = self._repeat_reference_vectors(n_pixels, imgs.device)
        return (sparse_pixels.reshape([n_pixels, self._reference_vectors.shape[0]]).unsqueeze(-1).repeat(1, 1, 3) * ref_vectors).sum(dim=1)


class NearestPointSparsifier(object):
    def __init__(self, n_channels, per_pixel_modules_distribution, sparsifying_rules, do_center=True):
        self._use_cache = False
        self._n_channels = n_channels
        self._per_pixel_modules_distribution = per_pixel_modules_distribution
        self._reference_points_by_module_ids = {}
        self._sparsifying_rules = []
        self._do_center = do_center
        if n_channels == -1:
            n_channels = 1
        self._norm_vec = torch.zeros([n_channels])
        for channel, module_id, reference_points, treat_zero_as_zero in sparsifying_rules:
            assert (self._per_pixel_modules_distribution == module_id).sum().item() == len(reference_points) + (0 if treat_zero_as_zero else 1)
            if 0 <= channel <= n_channels:
                self._sparsifying_rules.append(
                    (channel, module_id, SingleChannelPixelSparsifier(reference_points, treat_zero_as_zero),)
                )
                self._norm_vec[channel] += 1.0
            else:
                assert n_channels > 1
                assert channel == -1
                self._sparsifying_rules.append(
                    (channel, module_id, MultiChannelPixelSparsifier(reference_points, treat_zero_as_zero),)
                )
                self._norm_vec += 1.0
        self._straight_mappings = {}

    def set_use_cache(self, use_cache):
        self._use_cache = use_cache
        for _, _, pixel_sparsifier in self._sparsifying_rules:
            pixel_sparsifier.set_use_cache(use_cache)

    def _get_mapping(self, batch_size, height, width, module_id, device):
        if self._use_cache:
            key = (batch_size, height, width, module_id, device,)
            res = self._straight_mappings.get(key)
            if res is not None:
                return res
        else:
            key = None

        h_dim = self._per_pixel_modules_distribution.shape[0]
        w_dim = self._per_pixel_modules_distribution.shape[1]

        single_cell_tensor = torch.cartesian_prod(torch.arange(h_dim) * w_dim * width, torch.arange(w_dim)).sum(dim=-1)
        single_cell_tensor = single_cell_tensor[self._per_pixel_modules_distribution.flatten() == module_id].to(device=device)

        b_tensor = torch.arange(batch_size, device=device) * (height * h_dim * width * w_dim)
        i_tensor = torch.arange(height, device=device) * (width * w_dim * h_dim)
        j_tensor = torch.arange(width, device=device) * w_dim

        offsets = b_tensor.view(batch_size, 1, 1) + i_tensor.view(1, height, 1) + j_tensor.view(1, 1, width)
        res = (offsets.unsqueeze(-1) + single_cell_tensor).reshape(-1)

        if self._use_cache:
            self._straight_mappings[key] = res
        return res

    def __call__(self, imgs):
        if self._n_channels == -1:
            b, h, w = imgs.shape
        else:
            b, c, h, w = imgs.shape
            assert c == self._n_channels
        h_dim = self._per_pixel_modules_distribution.shape[0]
        w_dim = self._per_pixel_modules_distribution.shape[1]
        res = torch.zeros([b * h * h_dim * w * w_dim], device=imgs.device, dtype=torch.float32)
        for channel, module_id, pixel_sparsifier in self._sparsifying_rules:
            if self._n_channels == -1:
                c_imgs = imgs.reshape([b * h * w])
            else:
                if channel >= 0:
                    c_imgs = imgs[:, channel, :, :].reshape([b * h * w])
                else:
                    c_imgs = imgs.permute([0, 2, 3, 1]).reshape([b * h * w, self._n_channels])
            sparsified_pixels = pixel_sparsifier(c_imgs)
            mapping = self._get_mapping(b, h, w, module_id, imgs.device)
            res[mapping] = sparsified_pixels.flatten()
        if self._do_center:
            return (res.reshape([b, h * h_dim, w * w_dim]) - 0.5) * 2.0
        else:
            return res.reshape([b, h * h_dim, w * w_dim])

    def reverse(self, sparse_imgs):
        b, _h, _w = sparse_imgs.shape
        assert (_h % self._per_pixel_modules_distribution.shape[0]) == 0
        assert (_w % self._per_pixel_modules_distribution.shape[1]) == 0
        h = _h // self._per_pixel_modules_distribution.shape[0]
        w = _w // self._per_pixel_modules_distribution.shape[1]
        if self._n_channels == -1:
            res = torch.zeros([b * h * w], device=sparse_imgs.device, dtype=torch.float32)
        else:
            res = torch.zeros([b * h * w, self._n_channels], device=sparse_imgs.device, dtype=torch.float32)
        if self._do_center:
            sparse_imgs = ((sparse_imgs - 0.5) * 2.0).clip(-1.0)
            sparse_imgs = (sparse_imgs.reshape([b * _h * _w]) + 1.0) / 2.0
        else:
            sparse_imgs = sparse_imgs.reshape([b * _h * _w])
        for channel, module_id, pixel_sparsifier in self._sparsifying_rules:
            mapping = self._get_mapping(b, h, w, module_id, sparse_imgs.device)
            selected_modules = sparse_imgs[mapping]
            if self._n_channels == -1:
                res += pixel_sparsifier.reverse(selected_modules)
            else:
                if channel >= 0:
                    res[:, channel] += pixel_sparsifier.reverse(selected_modules)
                else:
                    res += pixel_sparsifier.reverse(selected_modules)

        self._norm_vec = self._norm_vec.to(device=sparse_imgs.device)
        if self._n_channels == -1:
            return (res / self._norm_vec).reshape([b, h, w])
        else:
            return (res / self._norm_vec).reshape([b, h, w, self._n_channels]).permute([0, 3, 1, 2])
