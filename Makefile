STYLE = customdocstyle.html
SRC = src/lib.rs

docs: $(STYLE) $(SRC)
	rustdoc --html-in-header $(STYLE) -L target/deps $(SRC)

.PHONY: docs
