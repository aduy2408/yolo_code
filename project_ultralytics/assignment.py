"""Assignment helpers for preserving P2 spatial top-k under slot expansion."""

from __future__ import annotations

from itertools import permutations

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


class CollisionPreservingTaskAlignedAssigner(TaskAlignedAssigner):
    """TAL with P2 top-k selected over spatial locations, not duplicate slots."""

    p2_base_count: int = 0
    p2_slot_count: int = 1

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        if self.p2_slot_count <= 1 or self.p2_base_count <= 0:
            return super()._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        batch, n_gt, _ = align_metric.shape
        p2_count = self.p2_base_count
        slots = self.p2_slot_count
        p2_align = align_metric[:, :, : slots * p2_count].view(batch, n_gt, slots, p2_count).permute(0, 1, 3, 2)
        p2_inside = (
            mask_in_gts[:, :, : slots * p2_count]
            .view(batch, n_gt, slots, p2_count)
            .permute(0, 1, 3, 2)
            .bool()
        )
        p2_location_metric = p2_align.masked_fill(~p2_inside, 0).amax(dim=-1)
        p2_topk = min(self.topk, p2_count)
        _, p2_indices = torch.topk(p2_location_metric, p2_topk, dim=-1, largest=True)
        p2_selected = torch.zeros_like(p2_location_metric, dtype=torch.bool)
        p2_selected.scatter_(2, p2_indices, True)
        p2_selected &= mask_gt.bool()

        mask_pos = torch.zeros_like(align_metric)
        non_p2 = align_metric[:, :, slots * p2_count :]
        if non_p2.shape[-1]:
            non_p2_topk = self.select_topk_candidates(
                non_p2,
                topk_mask=mask_gt.expand(-1, -1, self.topk).bool(),
            )
            mask_pos[:, :, slots * p2_count :] = non_p2_topk

        if slots == 2:
            # The experiment uses K=2. Select at most two GT identities per
            # location, then compare the two possible slot permutations in a
            # single tensor operation. This avoids synchronizing Python once
            # for each of the 25,600 P2 locations.
            candidate_scores = p2_location_metric.masked_fill(~p2_selected, float("-inf"))
            candidate_scores, candidate_gt = torch.topk(candidate_scores, 2, dim=1)
            candidate_scores = candidate_scores.permute(0, 2, 1)
            candidate_gt = candidate_gt.permute(0, 2, 1)
            candidate_valid = torch.isfinite(candidate_scores)

            candidate_align = p2_align.permute(0, 2, 1, 3).gather(
                2, candidate_gt.unsqueeze(-1).expand(-1, -1, -1, slots)
            )
            first_valid = candidate_valid[:, :, 0]
            second_valid = candidate_valid[:, :, 1]
            both_valid = first_valid & second_valid

            first_align = candidate_align[:, :, 0]
            second_align = candidate_align[:, :, 1]
            keep_score = first_align[..., 0] + second_align[..., 1]
            swap_score = first_align[..., 1] + second_align[..., 0]
            swap = both_valid & (swap_score > keep_score)

            first_slot = torch.where(
                both_valid,
                swap.long(),
                first_align.argmax(dim=-1),
            )
            second_slot = torch.where(swap, torch.zeros_like(first_slot), torch.ones_like(first_slot))

            batch_index = torch.arange(batch, device=align_metric.device)[:, None].expand(batch, p2_count)
            location = torch.arange(p2_count, device=align_metric.device)[None, :].expand(batch, p2_count)
            mask_pos[batch_index[first_valid], candidate_gt[:, :, 0][first_valid], location[first_valid] + first_slot[first_valid] * p2_count] = 1
            mask_pos[batch_index[second_valid], candidate_gt[:, :, 1][second_valid], location[second_valid] + second_slot[second_valid] * p2_count] = 1
        else:
            for batch_index in range(batch):
                for location in range(p2_count):
                    candidates = torch.where(p2_selected[batch_index, :, location])[0].tolist()
                    if not candidates:
                        continue
                    if len(candidates) > slots:
                        candidates = sorted(
                            candidates,
                            key=lambda gt: float(p2_location_metric[batch_index, gt, location]),
                            reverse=True,
                        )[:slots]
                    best_score = None
                    best_perm = None
                    for perm in permutations(range(slots), len(candidates)):
                        score = sum(
                            p2_align[batch_index, gt, location, slot] for gt, slot in zip(candidates, perm)
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_perm = perm
                    for gt, slot in zip(candidates, best_perm or ()):
                        mask_pos[batch_index, gt, location + slot * p2_count] = True

        mask_pos = mask_pos * mask_in_gts * mask_gt
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        overlaps *= mask_pos
        pos_overlaps = overlaps.amax(dim=-1, keepdim=True)
        align_metric.mul_(pos_overlaps).div_(pos_align_metrics + self.eps)
        norm_align_metric = align_metric.amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
