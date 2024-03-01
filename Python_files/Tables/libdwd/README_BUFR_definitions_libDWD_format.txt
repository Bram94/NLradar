====================================================================================================
                Formatbeschreibung der WMO Manual on Codes Dateien im libDWD-Format
====================================================================================================
Autor: florian.teichert@dwd.de
Version: 2012-07-12
Weitere Infos in der /dbDoku/ im MA-Portal
     IT/Messnetz/Technik -> Datenmanagement (technisch) ->
         Portlet: "Management der DWD Fachdaten - Dokumentation"

----- Verzeichnisse und Dateinamen -----------------------------------------------------------------
Dateien im Hauptverzeichnis beziehen sich auf WMO Ver�ffentlichungen. In den Unterverzeichnissen
finden sich nationale MoC-Erweiterungen. Die Verzeichnisnamen sind aus 'GenCentre' und
'GenSubCentre' zusammengesetzt, jeweils sechsstellig mit Vornullen.
Dateinamen sind einheitlich gebildet aus Inhalt und dreistelliger Versionsnummer mit Vornullen. Die
Dateien mit Erweiterung '.val' sind identisch mit den Dateien gleichen Dateinamens, enthalten aber
zus�tzlich Inhalte mit Status 'Validation'.

----- Dateiformat allgemein ------------------------------------------------------------------------
Alle Dateien sind im Format Unix-ASCII mit LineFeed als Zeilenumbruch. Die Spalten werden durch
Tabulator getrennt. Kommentarzeilen sind mit '#' an erster Stelle gekennzeichnet. Jede Datei wird
mit einem kurzen beschreibenden Kommentarkopf eingeleitet, Kommentare k�nnen aber auch innerhalb der
Datenzeilen auftreten.

----- Dateien: table_b_* ---------------------------------------------------------------------------
In diesen Dateien sind die Element-Deskriptoren des jeweiligen MoC enthalten und entsprechen damit
dem Inhalt der WMO Table B. Zeilenformat ist im Kommentarkopf beschrieben.
Hinweis: Die Spalte "libDWDType" kann folgende Werte enthalten:
    'A' f�r ASCII
    'C' f�r CodeTable
    'F' f�r FlagTable
    'N' f�r Numeric
    '?' f�r unbekannt, sollte als 'A' interpretiert werden
Siehe auch im MA-Portal unter: /dbDoku/ -> BUFR -> WMO Manual on Codes -> Einheiten

----- Dateien: table_d_* ---------------------------------------------------------------------------
In diesen Dateien sind die Sequenzen des jeweiligen MoC enthalten und entsprechen damit dem Inhalt
der WMO Table D. Zeilenformat ist im Kommentarkopf beschrieben.

----- Dateien: codeflags_* -------------------------------------------------------------------------
In diesen Dateien sind die Code- und FlagTables und des jeweiligen MoC enthalten und entsprechen
damit dem Inhalt der WMO CodeFlags. Zeilenformat ist im Kommentarkopf beschrieben.
Hinweis: Die Spalte "libDWDType" kann folgende Werte enthalten:
    'C' f�r CodeTable
    'F' f�r FlagTable
Hinweis: Die Spalte codeFigureFrom enth�lt immer einen Wert, wobei f�r libDWDType=C eine CodeFigure
         gemeint ist f�r libDWDType=F eine BitNo.
Hinweis: Die Spalte codeFigureTo enth�lt f�r libDWDType=C nur dann Werte, wenn ein Bereich von-bis
         beschrieben werden soll. F�r libDWDType=F wird in dieser Spalte ein 'A' angegeben, um
         'alle Bits gesetzt' zu codieren, womit �blicherweise 'missing value' gemeint ist.
====================================================================================================
