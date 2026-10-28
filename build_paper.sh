#!/bin/bash
cd /mnt/d/Axioms/lol-game-ml-prediction/paper/entropy_submission || exit 1
export TEXINPUTS='./Definitions:'
for n in 1 2 3; do
  pdflatex -interaction=nonstopmode main.tex > "c$n.log" 2>&1
done
echo "errors:   $(grep -c '^!' c3.log)"
echo "undefined: $(grep -ci 'undefined' c3.log)"
echo "overfull:  $(grep -c 'Overfull' c3.log)"
pdfinfo main.pdf 2>/dev/null | grep -E 'Pages|Page size'
rm -f c1.log c2.log c3.log main.aux main.out main.toc main.log
