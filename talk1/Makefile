
all:
	rustdoc index.md -o . --html-in-header=header.inc.html --markdown-no-toc

test: ndarray
	rustdoc --test -L ./ndarray/target/debug -L ./ndarray/target/debug/deps index.md --test-args --nocapture

ndarray:
	(test -d ndarray || git clone https://github.com/bluss/rust-ndarray ndarray)
	(cd ndarray; cargo build)

.PHONY: ndarray
