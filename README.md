# Sam-Jar Search

Forschungsfrage: Welchen Einfluss hat der Hyperparameter «window_size» des Word-Embedding Systems Doc2Vec auf die Retrieval-Effektivität eines Systems? Hierbei analysiert eine tiefe «window_size» eher die Austauschbarkeit eines Wortes mit einem anderen und eine hohe «window_size» eher den Kontext eines Wortes. Ist für Retrieval-Effektivität eines Systems die Austauschbarkeit oder der Kontext eines Wortes ausschlaggebend?

## Systeme/Aufbau
Für das Einlesen, Preprocessing, Indexieren, Trainieren des Models und die spätere Analyse wurde ein Pythonskript geschrieben. Dabei wurden hilfreiche Libraries wie spaCy fürs Preprocessing und GenSim fürs Doc2Vec-Model verwendet. Als erster Schritt werden alle Dokumente und Queries eingelesen. Die nlp-Funktion von spaCy führt das Preprocessing in mehreren Schritten durch. 
1.	Tokenisierung
2.	Parser: Labeln nach Wortart (Nomen, Verben, etc.) 
3.	Filtern: Alles ausser "NOUN", "ADJ", "VERB", "ADV" wird verworfen
4.	Lemmatisierung: Umformung in die Stammform

In unserem Beispiel wurde das erste Dokument mit der Form:”computation of arc tan n for using an electronic computers” in folgende Form umgewandelt: "computation use electronic computer”
Als zweiten Schritt werden die Dokumente getagged und anschliessend für das Trainieren des Doc2Vec-Model eingespiesen. Das Model behält dabei alle Standardparameter, ausser die “vector_size” wurde auf 200 gesetzt und die “epochs” auf 100. Zusätzlich wurde der Parameter “window” zu Analysezwecken angepasst. Wörter bekommen ihr Embedding, indem angeschaut wird, in welcher Reihenfolge sie neben anderen Wörtern vorkommen. Die Mechanik dabei ist die folgende. Es wird ein Fenster erzeugt, das sich über den gesamten Text verschiebt. Das gleitende Fenster erzeugt Trainingsmuster für unser Modell. Bei “window-size” N werden immer die letzten und die nächsten N Wörter für das Training verwendet.
Beispiel für window-size 1 am ersten Dokument ['digital', 'computer', 'university', 'part', 'reeve']

Grundsätzlich ist Doc2Vec (Paragraph Vector) ein Model, welches Dokumente als Vektoren darstellt. Es nutzt einen unsupervised Algorithmus der Merkmale mit fixer Länge aus Textstücken mit variabler Länge wie z.B. Sätze oder Dokumente lernt. Der Algorithmus repräsentiert jedes Dokument durch einen dichten Vektor, der darauf trainiert ist, Wörter im Dokument vorherzusagen.

## Queries
```txt
  <DOC>
    <recordId>5899</recordId>
    <text>simultaneous observations of whistlers and lightning
    discharges</text>
  </DOC>
  <DOC>
    <recordId>6324</recordId>
    <text>what does type compatibility mean in languages that allow
    programmer defined types? (you might want to restrict this to
    "extensible" languages that allow definition of abstract data
    types or programmer-supplied definitions of operators like *,
    +.)</text>
  </DOC>
  <DOC>
    <recordId>7328</recordId>
    <text>the effect of oxidation on circuit breaker
    contacts</text>
  </DOC>
```



## Documents
```txt
  <DOC>
    <recordId>8</recordId>
    <text>a history of public libraries in bolton from the
    beginnings to 1974. (ph.d thesis). an outline of the municipal
    development of bolton, lancashire, is followed by an account of
    all public libraries known to have existed in the town before
    the first public libraries act of 1850 in 1852. a detailed
    evaluative history of the rate-supported public library service
    is given, from the work of the first library committee in
    1852-53 to the retirement of the chief librarian in
    1974.</text>
  </DOC>
  <DOC>
    <recordId>11</recordId>
    <text>automated acquisitions systems' papers presented at the
    lita institute-part 2. presents the concluding papers from the
    lita institute on automated acquisitions. for abstracts of the
    papers presented see the following serial numbers'.</text>
  </DOC>
```

## Rang
```txt
245 Q0 14814 0 0.7877904772758484 Sam.jar
245 Q0 38008 1 0.7494387030601501 Sam.jar
245 Q0 622 2 0.7364974617958069 Sam.jar
245 Q0 30014 3 0.7234606742858887 Sam.jar
245 Q0 34188 4 0.7154969573020935 Sam.jar
245 Q0 1716 5 0.7018116116523743 Sam.jar
```
