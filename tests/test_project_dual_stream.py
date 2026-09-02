import torch

from project_ultralytics.modules import DualChannelFormationBackbone, DualCollapse, DualDownsample


def test_dual_stream_preserves_tuple_across_formation_and_downsample() -> None:
    x = torch.randn(1, 32, 32, 32)
    formation = DualChannelFormationBackbone(32, 32, n=1, progressive=True)
    streams = formation(x)
    assert isinstance(streams, tuple)
    assert len(streams) == 2
    assert streams[0].shape[1] + streams[1].shape[1] == 32

    downsampled = DualDownsample(32, 64)(streams)
    assert downsampled[0].shape[-2:] == (16, 16)
    assert downsampled[1].shape[-2:] == (16, 16)
    assert DualCollapse(64, 24)(downsampled).shape == (1, 24, 16, 16)
