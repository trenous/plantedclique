(TeX-add-style-hook
 "der"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "amsmath"
    "amssymb"
    "amsfonts")
   (TeX-add-symbols
    '("E" 1)
    "expectation"
    "mut"))
 :latex)

