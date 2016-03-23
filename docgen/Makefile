DOCCRATES = ndarray ndarray-rblas ndarray-rand

# deps to delete the generated docs
RMDOCS = 

FEATURES = ""

VERSIONS = $(patsubst %,target/VERS/%,$(DOCCRATES))

DESTDIR = ../master

docs: mkdocs subst $(RMDOCS)

# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
$(VERSIONS): Cargo.toml
	mkdir -p $(@D)
	cargo pkgid $(@F)  | sed -e "s/.*#\(\|.*:\)//" > "$@"

$(DOCCRATES): %: target/VERS/%
	# Put in the crate version into the docs
	find  $(DESTDIR)/$(subst -,_,$@) -name "*.html" -exec sed -i -e "s/<title>\(.*\) - Rust/<title>$@ $(shell cat $<) - \1 - Rust/g" {} \;

subst: $(DOCCRATES)

mkdocs: Cargo.toml
	for crate in $(DOCCRATES) ; do \
		cargo doc --no-deps -p $$crate ; \
	done
	rm -rf $(DESTDIR)
	cp -r ./target/doc $(DESTDIR)
	cp ./split_at.svg $(DESTDIR)/ndarray/
	- cat ./custom.css >> $(DESTDIR)/main.css

$(RMDOCS): mkdocs
	rm -r $(DESTDIR)/$@
	sed -i "/searchIndex\['$@'\]/d" $(DESTDIR)/search-index.js

fast: FEATURES = 
fast: mkdocs 

.PHONY: docs mkdocs subst $(DOCCRATES) $(RMDOCS)
