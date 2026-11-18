"""Conservative mask geometry checks, using only VLM boxes and trusted masks.

These checks reject obvious merges and jumps; they cannot prove semantic object
identity. A connected mask may still cover touching objects, so the task's VLM
mask confirmation remains necessary. Rejection means loss, never task success.
"""
from dataclasses import dataclass
import math

import numpy as np


def _components(mask, minimum):
    """Eight-connected components via row runs; no scipy/OpenCV dependency."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    parents = [0]

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    previous = []
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        current = []
        cursor = 0
        for x0, x1 in zip(edges[::2], edges[1::2]):
            while cursor < len(previous) and previous[cursor][1] < x0:
                cursor += 1
            adjacent = []
            for index in range(cursor, len(previous)):
                a, b, label = previous[index]
                if a > x1:
                    break
                adjacent.append(root(label))
            if adjacent:
                label = adjacent[0]
                for other in adjacent[1:]:
                    parents[root(other)] = root(label)
            else:
                label = len(parents)
                parents.append(label)
            labels[y, x0:x1] = label
            current.append((int(x0), int(x1), label))
        previous = current
    lookup = np.asarray([root(i) for i in range(len(parents))], dtype=np.int32)
    labels = lookup[labels]
    counts = np.bincount(labels.ravel(), minlength=len(parents))
    significant = np.flatnonzero(counts >= minimum)
    significant = significant[significant != 0]
    # Reject pathological/noisy outputs without allocating hundreds of masks.
    if len(significant) > 64:
        return labels, [], int(len(significant))
    components = []
    for label in significant:
        yy, xx = np.nonzero(labels == label)
        components.append(dict(label=int(label), pixels=int(len(xx)),
            bbox=[int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
            center=[float(xx.mean()), float(yy.mean())]))
    return labels, components, len(components)


def _geometry(mask):
    yy, xx = np.nonzero(mask)
    return dict(pixels=int(len(xx)), bbox=[int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
                center=[float(xx.mean()), float(yy.mean())])


@dataclass
class GuardResult:
    mask: np.ndarray
    metadata: dict

    @property
    def accepted(self):
        return self.metadata['accepted']


class SingleTargetGuard:
    def __init__(self, min_pixels=4):
        self.min_pixels = int(min_pixels)
        self.last_trusted = None
        self.fuller_anchor = None
        self.peak_trusted_pixels = 0

    def trust(self, mask, reset_anchor=False):
        trusted = np.asarray(mask, dtype=bool).copy()
        self.last_trusted = trusted
        pixels = int(trusted.sum())
        if reset_anchor or self.fuller_anchor is None or self.fuller_anchor.shape != trusted.shape:
            self.peak_trusted_pixels = pixels
            self.fuller_anchor = trusted.copy()
        else:
            self.peak_trusted_pixels = max(self.peak_trusted_pixels, pixels)
            # Progressive occlusion must not shrink the size/shape reference
            # until a tiny visible fragment becomes the only recoverable size.
            if pixels >= .8*self.peak_trusted_pixels:
                self.fuller_anchor = trusted.copy()

    def _result(self, mask, accepted, reason, **details):
        return GuardResult(np.asarray(mask, dtype=bool).copy() if accepted else np.zeros(mask.shape, dtype=bool),
                           dict(enabled=True, accepted=bool(accepted), reason=reason, **details))

    def seed(self, mask, bbox):
        mask = np.asarray(mask, dtype=bool)
        x0, y0, x1, y1 = bbox
        reference_area = (x1-x0)*(y1-y0)
        minimum = max(self.min_pixels, math.ceil(reference_area*.08))
        labels, components, count = _components(mask, minimum)
        details = dict(significant_components=count, minimum_component_pixels=minimum,
                       tight_bbox_xyxy=list(bbox))
        if count > 64:
            return self._result(mask, False, 'too_many_components', **details)
        left, top = max(0, math.floor(x0)), max(0, math.floor(y0))
        right, bottom = min(mask.shape[1], math.ceil(x1)), min(mask.shape[0], math.ceil(y1))
        supported = []
        for component in components:
            overlap = int(np.count_nonzero(labels[top:bottom, left:right] == component['label']))
            component['bbox_overlap_pixels'] = overlap
            component['bbox_support_fraction'] = overlap/component['pixels']
            if overlap >= self.min_pixels and component['bbox_support_fraction'] >= .25:
                supported.append(component)
        details['components'] = components
        details['supported_components'] = len(supported)
        if len(supported) != 1:
            return self._result(mask, False, 'ambiguous_seed_components' if len(supported) > 1
                                else 'no_component_supported_by_tight_bbox', **details)
        component = supported[0]
        if component['bbox_support_fraction'] < .5:
            return self._result(mask, False, 'seed_extends_beyond_tight_bbox', **details)
        selected = labels == component['label']
        details['selected_pixels'] = component['pixels']
        details['discarded_pixels'] = int(mask.sum())-component['pixels']
        return self._result(selected, True, 'unique_tight_bbox_component', **details)

    def prediction(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if self.last_trusted is None or not self.last_trusted.any():
            return self._result(mask, False, 'no_trusted_seed')
        old = self.last_trusted
        if old.shape != mask.shape:
            return self._result(mask, False, 'mask_shape_changed')
        reference = _geometry(old)
        anchor = self.fuller_anchor if self.fuller_anchor is not None else old
        fuller = _geometry(anchor)
        minimum = max(self.min_pixels, math.ceil(reference['pixels']*.08))
        labels, components, count = _components(mask, minimum)
        details = dict(significant_components=count, minimum_component_pixels=minimum,
                       trusted_pixels=reference['pixels'], anchor_pixels=fuller['pixels'],
                       peak_trusted_pixels=self.peak_trusted_pixels,
                       reference_used='recent_trusted')
        if not components or count > 64:
            return self._result(mask, False, 'too_many_components' if count > 64 else 'no_significant_component', **details)
        selected = np.isin(labels, [c['label'] for c in components])
        current = _geometry(selected)
        area_ratio = current['pixels']/reference['pixels']
        distance = math.dist(current['center'], reference['center'])
        old_box, box = reference['bbox'], current['bbox']
        old_diag = math.hypot(old_box[2]-old_box[0], old_box[3]-old_box[1])
        new_diag = math.hypot(box[2]-box[0], box[3]-box[1])
        allowance = max(8., .75*old_diag)
        details.update(area_ratio=area_ratio, centroid_shift_pixels=distance,
                       allowed_shift_pixels=allowance, selected_pixels=current['pixels'])
        if len(components) > 1:
            # Occlusion may split one silhouette. Keep fragments only when
            # every significant piece overlaps the previous trusted silhouette,
            # and their union stays within its compact extent. A second object
            # elsewhere is rejected rather than choosing whichever is largest.
            fractions = [int(np.count_nonzero((labels == c['label']) & old))/c['pixels'] for c in components]
            compact = (box[0] >= old_box[0]-2 and box[1] >= old_box[1]-2
                       and box[2] <= old_box[2]+2 and box[3] <= old_box[3]+2)
            details['trusted_overlap_fractions'] = fractions
            if not (compact and area_ratio <= 1.25 and all(f >= .5 for f in fractions)):
                return self._result(mask, False, 'multiple_objects_or_discontinuous_fragments', **details)
        area_ok = .2 <= area_ratio <= 1.8
        motion_ok = distance <= allowance and new_diag <= 2*old_diag
        if (count == 1 and reference['pixels'] < .8*self.peak_trusted_pixels
                and not (area_ok and motion_ok)):
            # A fuller single-object silhouette may reappear after occlusion.
            # Check both its remembered size/location and its proximity to the
            # recent fragment; never rescue a second component or a far object.
            anchor_box = fuller['bbox']
            anchor_diag = math.hypot(anchor_box[2]-anchor_box[0], anchor_box[3]-anchor_box[1])
            anchor_ratio = current['pixels']/fuller['pixels']
            anchor_distance = math.dist(current['center'], fuller['center'])
            intersection = int(np.count_nonzero(selected & anchor))
            anchor_iou = intersection/(current['pixels']+fuller['pixels']-intersection)
            recovery_shift = max(8., .75*anchor_diag)
            recovered = (.2 <= anchor_ratio <= 1.8 and distance <= recovery_shift
                         and anchor_distance <= max(8., anchor_diag) and anchor_iou >= .05
                         and new_diag <= 2*anchor_diag)
            details.update(anchor_area_ratio=anchor_ratio, anchor_centroid_shift_pixels=anchor_distance,
                           anchor_iou=anchor_iou, recovery_allowed_shift_pixels=recovery_shift,
                           anchor_recovery_accepted=bool(recovered))
            if recovered:
                details['reference_used'] = 'fuller_anchor'
                return self._result(selected, True, 'recovered_single_target_after_occlusion', **details)
        if not area_ok:
            return self._result(mask, False, 'area_changed_abruptly', **details)
        if not motion_ok:
            return self._result(mask, False, 'spatial_identity_jump', **details)
        reason = 'trusted_silhouette_fragments' if count > 1 else 'continuous_single_component'
        return self._result(selected, True, reason, **details)
