DOCCRATES = ndarray

# deps to delete the generated docs
RMDOCS =

FEATURES = "assign_ops rustc-serialize rblas"

VERSIONS = $(patsubst %,target/VERS/%,$(DOCCRATES))

docs: mkdocs subst $(RMDOCS)

# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
$(VERSIONS): Cargo.toml
	mkdir -p $(@D)
	cargo pkgid ndarray | sed -e "s/.*#\(\|.*:\)//" > "$@"

$(DOCCRATES): %: target/VERS/%
	# Put in the crate version into the docs
	find ./master/$@ -name "*.html" -exec sed -i -e "s/<title>\(.*\) - Rust/<title>ndarray $(shell cat $<) - \1 - Rust/g" {} \;

subst: $(DOCCRATES)

mkdocs: Cargo.toml
	cargo doc --no-deps --features=$(FEATURES)
	rm -rf ./master
	cp -r ./target/doc ./master
	- cat ./custom.css >> master/main.css

$(RMDOCS): mkdocs
	rm -r ./master/$@
	sed -i "/searchIndex\['$@'\]/d" master/search-index.js

fast: FEATURES = 
fast: mkdocs 

.PHONY: docs mkdocs subst $(DOCCRATES) $(RMDOCS)
