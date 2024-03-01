====================================================================================================
                Formatbeschreibung der WMO Manual on Codes Dateien im libDWD-Format
====================================================================================================
Autor: florian.teichert@dwd.de
Version: 2012-07-12
Weitere Infos in der /dbDoku/ im MA-Portal
     IT/Messnetz/Technik -> Datenmanagement (technisch) ->
         Portlet: "Management der DWD Fachdaten - Dokumentation"

----- Verzeichnisse und Dateinamen -----------------------------------------------------------------
Dateien im Hauptverzeichnis beziehen sich auf WMO Veröffentlichungen. In den Unterverzeichnissen
finden sich nationale MoC-Erweiterungen. Die Verzeichnisnamen sind aus 'GenCentre' und
'GenSubCentre' zusammengesetzt, jeweils sechsstellig mit Vornullen.
Dateinamen sind einheitlich gebildet aus Inhalt und dreistelliger Versionsnummer mit Vornullen. Die
Dateien mit Erweiterung '.val' sind identisch mit den Dateien gleichen Dateinamens, enthalten aber
zusätzlich Inhalte mit Status 'Validation'.

----- Dateiformat allgemein ------------------------------------------------------------------------
Alle Dateien sind im Format Unix-ASCII mit LineFeed als Zeilenumbruch. Die Spalten werden durch
Tabulator getrennt. Kommentarzeilen sind mit '#' an erster Stelle gekennzeichnet. Jede Datei wird
mit einem kurzen beschreibenden Kommentarkopf eingeleitet, Kommentare können aber auch innerhalb der
Datenzeilen auftreten.

----- Dateien: table_b_* ---------------------------------------------------------------------------
In diesen Dateien sind die Element-Deskriptoren des jeweiligen MoC enthalten und entsprechen damit
dem Inhalt der WMO Table B. Zeilenformat ist im Kommentarkopf beschrieben.
Hinweis: Die Spalte "libDWDType" kann folgende Werte enthalten:
    'A' für ASCII
    'C' für CodeTable
    'F' für FlagTable
    'N' für Numeric
    '?' für unbekannt, sollte als 'A' interpretiert werden
Siehe auch im MA-Portal unter: /dbDoku/ -> BUFR -> WMO Manual on Codes -> Einheiten

----- Dateien: table_d_* ---------------------------------------------------------------------------
In diesen Dateien sind die Sequenzen des jeweiligen MoC enthalten und entsprechen damit dem Inhalt
der WMO Table D. Zeilenformat ist im Kommentarkopf beschrieben.

----- Dateien: codeflags_* -------------------------------------------------------------------------
In diesen Dateien sind die Code- und FlagTables und des jeweiligen MoC enthalten und entsprechen
damit dem Inhalt der WMO CodeFlags. Zeilenformat ist im Kommentarkopf beschrieben.
Hinweis: Die Spalte "libDWDType" kann folgende Werte enthalten:
    'C' für CodeTable
    'F' für FlagTable
Hinweis: Die Spalte codeFigureFrom enthält immer einen Wert, wobei für libDWDType=C eine CodeFigure
         gemeint ist für libDWDType=F eine BitNo.
Hinweis: Die Spalte codeFigureTo enthält für libDWDType=C nur dann Werte, wenn ein Bereich von-bis
         beschrieben werden soll. Für libDWDType=F wird in dieser Spalte ein 'A' angegeben, um
         'alle Bits gesetzt' zu codieren, womit üblicherweise 'missing value' gemeint ist.
====================================================================================================
