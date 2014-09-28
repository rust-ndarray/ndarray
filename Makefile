STYLE = customdocstyle.html
SRC = src/lib.rs

NUM = --extern num=$(wildcard ./target/deps/libnum*)

docs: $(STYLE) $(SRC)
	rustdoc --html-in-header $(STYLE) -L target/deps $(NUM) $(SRC)

.PHONY: docs
