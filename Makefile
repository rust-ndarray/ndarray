
all: docs

%:
	make -C docgen -f Makefile $@

.PHONY: all
