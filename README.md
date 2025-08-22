# ZetaSketch RS

A native Rust re-implementation of the
[Zetasketch](https://github.com/google/zetasketch) Java library by Google.

The ZetaSketch library provides an implementation of the HyperLogLog++
algorithm used by several Google Cloud products, most notably BigQuery and
BigTable.

This library allows deserializing, modifying and serializing the HyperLogLog++
sketches used by these products.

You can learn more details about ZetaSketch and its HyperLogLog++
implementation in the
[README in the original
library](https://github.com/google/zetasketch/blob/master/README.md).

This reimplementation is based on version 0.1.0 of the original library, which
is the only version published.

## Compatibility

This library was implemented as a translation of the original Java code into
Rust, then refined here and there to improve performance and to be more
idiomatic. It strives to be 100% compatible with the Java library, which means
that for identical inputs, it should produce 100% identical sketches as the
Java library. Any deviation should be considered a bug.

We are using the [`j4rs`](https://github.com/astonbitecode/j4rs) crate for testing,
which allows to call us the original Java library from our Rust tests and compare
the behavior of both libraries.

## Python Bindings

The project also contains Python bindings for the Rust crate published on Pypi as
[`zetasketch-py`](https://pypi.org/project/zetasketch-py/).

The bindings are generated using [`maturin`](https://github.com/PyO3/maturin).

## Sponsorship

Porting the code over from Java to Rust was a fair bit of work that took me a fair
bit of time. If you find this library useful, please consider donating to help me
work on other similar projects in the future. Thank you!

## License

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file
for details.

The original library is licensed under the Apache License 2.0.

> [!NOTE]
> This project is in no way affiliated with or endorsed by Google.
