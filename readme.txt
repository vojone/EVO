# EVO - Design of image noise filters with CGP

## Author

Vojtěch Dvořák (xdvora3o)

## Popis

Jedná se o rámcové řešení pro provádění experimentů s CGP obrazových filtrů implementované v jazyce Python s využitím modulu hal-cgp.
Kromě Skriptu pro trénování obrazových filtrů pomocí CGP, archiv obsahuje také skripty pro analýzu výsledků, pro tvorbu trénovacích a testovaích dat, výsledky experimentů a jejich analýzu. 

## Obsah archivu

`data` - Trénovací a testovací data pro CGP

`experiments` - Popisy experimentů ve formátu JSON

`results` - Výsledky experimentů (podadresáře obsahují logy ve formátu .csv, parametry daného experimentu v .json, a nejlepší nalezené řešení ve formátu .pkl)

`src` - Zdrojové kódy

    `analysis.ipynp` - Analýza výsledků

    `analyze_results.py` - Pomocné funkce pro analýzu

    `common.py` - Pomocné funkce pro CGP

    `filter_cgp.py` - Rámec pro provádění experimentů s CGP

    `noise_images.py` - Skript pro vytváření trénovacích a testovaích dat z libovolných obrázků 

    `use_filter.py` - Skript pro snazší aplikace natrénovaného filtru na obrázek


`bootstrap.sh` - Bash skript pro instalaci závislostí  

`prezentace.pdf` - Prezentace s výsledky experimentů

## Instalace a spuštění 

Závislosti lze nainstalovat pomocí přiloženého skriptu `bootstrap.sh`, je však zapotřebí mít nainstalovaný `git`, `python`, `virtualenv` a správce balíků `pip`.
Ostatní závislosti jsou ale nainstalovány automaticky a je vytvořeno standardní virtuální prostředí pro jazyk Python.

```
. ./bootstrap.sh
```

Jednotlivé skripty ze složky `src` lze dále v rámci toho prostředí libovolně spouštět.
Stručný návod na použití je obsažen v hlavičcezdorojových souborů skriptů.
