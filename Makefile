#STYLE = customdocstyle.html
SRC = src/lib.rs

docs: $(STYLE) $(SRC)
	cargo doc --no-deps
	rm -r ./doc
	cp -r ./target/doc ./doc

.PHONY: docs
