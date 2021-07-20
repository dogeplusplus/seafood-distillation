import pytest

from models.convolutional import BaseCNN


def test_base_cnn_can_construct():
    filters = [1, 2, 3]
    kernel_sizes = [2, 2, 2]
    strides = [1, 1, 1]

    BaseCNN(filters, kernel_sizes, strides)


@pytest.mark.parametrize(
    "filters,kernel_sizes,strides",
    [
        ([1], [1, 2], [1]),
        ([1], [1], [1, 2]),
        ([1, 2], [1], [1])
    ])
def test_assertion_raised_unequal_lengths(filters, kernel_sizes, strides):
    with pytest.raises(Exception):
        BaseCNN(filters, kernel_sizes, strides)
