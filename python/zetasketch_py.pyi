from typing_extensions import TypeVar, Generic

T = TypeVar("T", str, bytes, int)

class HyperLogLogPlusPlus(Generic[T]):
    """
    HLL++ aggregator for estimating cardinalities of multisets.

    The aggregator uses the standard format for storing the internal state of the cardinality
    estimate as defined in hllplus-unique.proto, allowing users to merge aggregators with data
    computed in C++ or Go and to load up the cardinalities in a variety of analysis tools.

    The precision defines the accuracy of the HLL++ aggregator at the cost of the memory used. The
    upper bound on the memory required is 2<sup>precision</sup> bytes, but less memory is used for
    smaller cardinalities (up to ~2<sup>precision - 2</sup>). The relative error is 1.04 /
    sqrt(2<sup>precision</sup>). A typical value used at Google is 15, which gives an error of about
    0.6% while requiring an upper bound of 32 KiB of memory.
    """

    normal_precision: int
    """ Normal precision of the aggregator (default: 15). """

    sparse_precision: int | None
    """ Sparse precision of the aggregator or None if sparse representation is disabled (default: 20). """

    def __init__(
        self,
        ty: type[T],
        normal_precision: int = 14,
        sparse_precision: int | None = 12,
    ) -> None:
        """
        Create a new HyperLogLogPlusPlus aggregator.

        Args:
            ty: The type of the values to be added to the aggregator (str, bytes or int)
            normal_precision: The normal precision of the aggregator (default: 15).
            sparse_precision: The sparse precision of the aggregator (default: 20), set to None to disable sparse representation.
        """

    @staticmethod
    def from_bytes(data: bytes) -> "HyperLogLogPlusPlus[T]":
        """
        Create a new HyperLogLogPlusPlus aggregator from a serialized state.

        Use `to_bytes` to serialize the aggregator.

        Args:
            data: The serialized state of the aggregator.
        """

    def add(self, value: T) -> None:
        """
        Add a value to the aggregator.

        Args:
            value: The value to add to the aggregator.
        """

    def merge(self, other: "HyperLogLogPlusPlus[T]" | bytes) -> None:
        """
        Merge another HyperLogLogPlusPlus aggregator with this one.

        If a bytes object is provided, it is deserialized and merged
        with this aggregator.

        The other aggregator must be of the same type as this one.

        Args:
            other: The other HyperLogLogPlusPlus aggregator object or serialized
                   aggregator state to merge with this one.
        """

    def result(self) -> int:
        """
        Get the estimated cardinality of the aggregator.
        """

    def num_values(self) -> int:
        """
        Get the number of values added to the aggregator.
        """

    def to_bytes(self) -> bytes:
        """
        Get the serialized state of the aggregator.

        Use `from_bytes` to re-construct the aggregator from the serialized state.
        """

    def __repr__(self) -> str:
        """
        Get the string representation of the aggregator.
        """
