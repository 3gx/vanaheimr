
IMAGES = 

all: vanaheimr.pdf

vanaheimr.pdf: vanaheimr.tex vanaheimr.bib $(IMAGES)
	pdflatex -halt-on-error vanaheimr.tex
	bibtex vanaheimr
	pdflatex -halt-on-error vanaheimr.tex
	pdflatex -halt-on-error vanaheimr.tex
	
clean:
	rm -f *.pdf
	rm -f *.aux
	rm -f *.ps
	rm -f *.dvi
	rm -f *.log
	rm -f *.bbl
	rm -f *.blg	
	rm -f *.nav	
	rm -f *.out	
	rm -f *.snm	
	rm -f *.toc
