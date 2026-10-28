#!/bin/bash
cd /mnt/d/Axioms/lol-game-ml-prediction || exit 1
python3 build_supplementary.py || exit 1
cd paper/entropy_submission || exit 1
for n in 1 2 3; do
  pdflatex -interaction=nonstopmode supplementary.tex > "s$n.log" 2>&1
done
echo "errors:   $(grep -c '^!' s3.log)"
echo "undefined: $(grep -ci 'undefined' s3.log)"
echo "overfull:  $(grep -c 'Overfull' s3.log)"
echo "underfull: $(grep -c 'Underfull' s3.log)"
pdfinfo supplementary.pdf 2>/dev/null | grep -E 'Pages|Page size'
grep -E '^!' s3.log | head -8
rm -f s1.log s2.log s3.log supplementary.aux supplementary.out supplementary.log
