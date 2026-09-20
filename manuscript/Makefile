EVID  := ../evidence
CODE  := $(EVID)/code
PY    := python3

.PHONY: all evidence numbers figs clean check

all: main.pdf

## Re-run the whole analysis chain and refresh every generated artifact.
evidence:
	cd $(CODE) && $(PY) analyze.py
	cd $(CODE) && $(PY) make_figures.py
	cd $(CODE) && $(PY) make_tables.py
	cd $(CODE) && $(PY) export_numbers.py
	cd $(CODE) && $(PY) make_manifest.py
	$(MAKE) figs

numbers:
	cd $(CODE) && $(PY) export_numbers.py
	cd $(CODE) && $(PY) make_manifest.py

figs:
	cp $(EVID)/figures/*.pdf figs/
	cp $(EVID)/derived/tables/*.tex figs/

main.pdf: main.tex numbers.tex refs.bib
	tectonic -X compile main.tex --outfmt pdf --keep-logs
	$(PY) check_build.py

check:
	$(PY) check_build.py

clean:
	rm -f main.pdf main.log main.aux main.bbl main.blg main.out
