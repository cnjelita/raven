SRCFILE = raven_tests_doc
LATEX_FLAGS=-interaction=nonstopmode
TESTS_FILES = raven_tests_doc.tex test_doc_entries.tex

#all: raven_user_manual.pdf
all: tidy

tidy: $(TESTS_FILES)

	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex

draft: $(TESTS_FILES)

	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	bibtex $(SRCFILE)
	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf

../version.tex : $(MANUAL_FILES)
	git log -1 --format="%H %an %aD" .. > ../version.tex
